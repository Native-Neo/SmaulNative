#!/usr/bin/env python3
"""Embedding storage backends with a common interface.

``ram``  -- normal ``nn.Embedding`` (existing behavior, full table in RAM).
``mmap`` -- embedding table lives in a dedicated binary file (raw float32,
vocab*d, row-major) accessed through OS memory mapping. Only touched pages
are faulted in; the file is NEVER bulk-loaded into RAM.

Both expose the same interface so neural-network code needs no per-storage
branches: ``forward(ids)``, ``.weight`` (nn.Parameter), ``weight_numel()``,
``weight_nbytes()``, ``is_mmap``.

mmap details
------------
* File layout: raw little-endian float32, shape (vocab, d), row-major, no
  header. ``.nbytes == vocab*d*4`` exactly.
* Access: ``numpy.memmap(path, dtype=np.float32, mode='r+', shape=(v, d))``
  shared (zero-copy) with ``torch.from_numpy``. Lookup gathers only the
  requested rows; row granularity keeps random access efficient.
* Training: ``weight`` is a real ``nn.Parameter`` sharing the mapping, so the
  existing optimizer works unchanged and updates write through to the file.
  Optimizer momentum/grad buffers stay in normal RAM by design (only the
  embedding table itself is mapped).
* Device: the mapped weight always stays on CPU fp32 (like the RAM
  embedding's fp32 table). If lookup ids live on CUDA, gather on CPU then
  move only the looked-up rows to the device.

Limitations (honest): grad/momentum for the embedding remain dense RAM
buffers; peak-RAM savings equal roughly one embedding table, not the full
optimizer state. CUDA runs still copy looked-up rows to GPU per step.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_FILE = "embeddings.dat"


class RamEmbedding(nn.Module):
    is_mmap = False

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError(f"vocab/d must be positive, got {vocab_size}/{d_model}")
        self.emb = nn.Embedding(vocab_size, d_model)
        with torch.no_grad():
            # Same init as the historical SmaulLinear embedding (FP32, N(0, 0.02)).
            self.emb.weight.data.normal_(0, 0.02)

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, idx):
        return self.emb(idx)

    def weight_numel(self) -> int:
        return self.emb.weight.numel()

    def weight_nbytes(self) -> int:
        return self.emb.weight.numel() * 4


class MmapEmbedding(nn.Module):
    is_mmap = True

    def __init__(self, vocab_size: int, d_model: int, path=None):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError(f"vocab/d must be positive, got {vocab_size}/{d_model}")
        import numpy as np

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        # Whether this file is one we just created. Freshness used to be decided
        # by sampling row 0 and the last row and checking they were zero, which
        # meant a real checkpoint whose first and last rows happened to be all
        # zeros had its entire table overwritten with N(0, 0.02) -- silently,
        # with no warning and nothing to notice downstream except that the model
        # had stopped working. Provenance is knowable; content is not.
        fresh = True
        # Whether the file below is ours to delete: a path=None construction
        # owns its mkstemp file (consumed by __del__); an explicit path --
        # usually the checkpoint -- is the caller's.
        self._owns_file = path is None
        if path is None:
            fd, tmp = tempfile.mkstemp(prefix="smaul_emb_", suffix=".dat")
            os.close(fd)
            # Pre-size the file; memmap requires the full extent to exist.
            with open(tmp, "wb") as f:
                f.truncate(self.vocab_size * self.d_model * 4)
            self.path = Path(tmp)
        else:
            self.path = Path(path)
            if self.path.exists():
                fresh = False
                if self.path.stat().st_size != self.vocab_size * self.d_model * 4:
                    raise ValueError(
                        f"mmap file {self.path} is {self.path.stat().st_size} bytes, "
                        f"expected {self.vocab_size * self.d_model * 4}")
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "wb") as f:
                    f.truncate(self.vocab_size * self.d_model * 4)
        # TRUE mmap: shares file pages, never bulk-loads the file into RAM, and
        # writes through; only touched pages are faulted in.
        self._mem = np.memmap(str(self.path), dtype=np.float32, mode="r+",
                              shape=(self.vocab_size, self.d_model))
        tensor = torch.from_numpy(self._mem)
        # Only a file we just created is zero-filled, so give it the standard
        # N(0, 0.02) init with chunked writes (never materializing a second full
        # copy). An existing file is whatever it says it is and is left alone.
        if fresh:
            with torch.no_grad():
                for r0 in range(0, self.vocab_size, 1024):
                    r1 = min(self.vocab_size, r0 + 1024)
                    tensor[r0:r1].normal_(0, 0.02)
                self._mem.flush()
        self.weight = nn.Parameter(tensor)

    def _apply(self, fn, recurse=True):
        # Deliberately does not call nn.Module._apply: the mapped weight must
        # stay on CPU fp32, and a .to(device) or .half() must not copy the
        # table off its mapping. This module has no children today, so
        # recursing is a no-op kept only for future submodules.
        return self

    def forward(self, idx):
        w = self.weight
        if idx.device != w.device:
            # Row-based: gather on CPU (faults only needed pages), move only
            # the looked-up rows to the target device.
            out = F.embedding(idx.cpu(), w.cpu())
            return out.to(idx.device)
        return F.embedding(idx, w)

    def weight_numel(self) -> int:
        return self.vocab_size * self.d_model

    def weight_nbytes(self) -> int:
        return self.vocab_size * self.d_model * 4

    def flush(self) -> None:
        try:
            self._mem.flush()
        except (AttributeError, ValueError):
            pass

    def __del__(self):
        try:
            self.flush()
            # Temp files we created (path=None) are unlinked here: nothing
            # else knows their names, so without this every construction leaks
            # one vocab*d*4-byte file in /tmp. Explicit-path files are the
            # caller's (often the checkpoint) and are left alone.
            if getattr(self, "_owns_file", False):
                try:
                    del self._mem
                except AttributeError:
                    pass
                os.unlink(self.path)
        except Exception:
            pass


def create_embedding(vocab_size: int, d_model: int, storage: str = "ram",
                     path=None) -> nn.Module:
    if storage == "ram":
        return RamEmbedding(vocab_size, d_model)
    if storage == "mmap":
        return MmapEmbedding(vocab_size, d_model, path)
    raise ValueError(f"unknown embedding storage {storage!r}; expected 'ram' or 'mmap'")


def copy_embedding_to_mmap(ram_weight: torch.Tensor, path: Path,
                           chunk: int = 1024) -> None:
    """Chunked RAM -> mmap-file copy (never holds two full tables in RAM)."""
    import numpy as np

    v, d = ram_weight.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(v * d * 4)
    mem = np.memmap(str(path), dtype=np.float32, mode="r+", shape=(v, d))
    w = ram_weight.detach().cpu().float()
    with torch.no_grad():
        for r0 in range(0, v, chunk):
            mem[r0:min(v, r0 + chunk)] = w[r0:min(v, r0 + chunk)].numpy()
    mem.flush()
    del mem
