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
        if path is None:
            fd, tmp = tempfile.mkstemp(prefix="smaul_emb_", suffix=".dat")
            os.close(fd)
            # Pre-size the file; memmap requires the full extent to exist.
            with open(tmp, "wb") as f:
                f.truncate(self.vocab_size * self.d_model * 4)
            self.path = Path(tmp)
            self._owns_file = True
        else:
            self.path = Path(path)
            self._owns_file = False
            if self.path.exists():
                if self.path.stat().st_size != self.vocab_size * self.d_model * 4:
                    raise ValueError(
                        f"mmap file {self.path} is {self.path.stat().st_size} bytes, "
                        f"expected {self.vocab_size * self.d_model * 4}")
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "wb") as f:
                    f.truncate(self.vocab_size * self.d_model * 4)
        # TRUE mmap: shares file pages, never bulk-loads the file into RAM.
        # writes through to the file; only touched pages are faulted in.
        import numpy as np  # noqa: F811

        self._mem = np.memmap(str(self.path), dtype=np.float32, mode="r+",
                              shape=(self.vocab_size, self.d_model))
        tensor = torch.from_numpy(self._mem)
        # Fresh files read as zeros; give them the standard N(0, 0.02) init
        # with chunked writes (never materializing a second full copy).
        # Only sample edge rows to decide: scanning the whole mapping here
        # would fault every page into RAM and defeat lazy paging.
        _sample = tensor[:1].clone().detach()
        _tail = tensor[-1:].clone().detach()
        if bool((_sample == 0).all()) and bool((_tail == 0).all()):
            with torch.no_grad():
                for r0 in range(0, self.vocab_size, 1024):
                    r1 = min(self.vocab_size, r0 + 1024)
                    tensor[r0:r1].normal_(0, 0.02)
                self._mem.flush()
        self.weight = nn.Parameter(tensor)

    def _apply(self, fn, recurse=True):
        # Keep the mapped weight on CPU fp32: device/dtype moves apply to any
        # future buffers but must not copy the table off its mapping.
        if recurse:
            for module in self.children():
                module._apply(fn)
        # Intentionally skip self.weight (the mapping). Everything else in
        # this module (nothing today) would use the default path.
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
