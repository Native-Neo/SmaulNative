import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from embeddings import MmapEmbedding, RamEmbedding, copy_embedding_to_mmap, create_embedding


def _write(path, v, d, data):
    mem = np.memmap(str(path), dtype=np.float32, mode="w+", shape=(v, d))
    mem[:] = data
    mem.flush()
    del mem
    return data


# --- the two backends agree, which is the whole point of the abstraction ----

def test_both_backends_expose_the_same_interface():
    for backend in ("ram", "mmap"):
        emb = create_embedding(32, 8, backend)
        assert emb.is_mmap == (backend == "mmap")
        for name in ("weight", "forward", "weight_numel", "weight_nbytes"):
            assert hasattr(emb, name), (backend, name)
        assert emb.weight_numel() == 32 * 8
        assert emb.weight_nbytes() == 32 * 8 * 4


def test_forward_gathers_rows_in_both_backends():
    ids = torch.tensor([0, 3, 7, 3])
    ram = RamEmbedding(32, 8)
    with torch.no_grad():
        ram.weight.copy_(torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8))
    assert torch.equal(ram(ids), torch.nn.functional.embedding(ids, ram.weight))


def test_mmap_forward_matches_its_own_weight():
    emb = MmapEmbedding(16, 4)
    ids = torch.tensor([0, 5, 15, 5])
    assert torch.equal(emb(ids), torch.nn.functional.embedding(ids, emb.weight.detach()))


def test_the_two_backends_produce_identical_given_the_same_weights(tmp_path):
    """A backend must not change what the model computes."""
    v, d = 24, 6
    data = torch.randn(v, d)
    ram = RamEmbedding(v, d)
    with torch.no_grad():
        ram.weight.copy_(data)
    path = tmp_path / "e.dat"
    copy_embedding_to_mmap(data, path)
    mmap = MmapEmbedding(v, d, path)
    ids = torch.randint(0, v, (3, 11))
    assert torch.allclose(ram(ids), mmap(ids), atol=1e-6)


def test_both_backends_validate_their_shape():
    for backend in ("ram", "mmap"):
        for v, d in ((0, 8), (8, 0), (-1, 8), (8, -3)):
            with pytest.raises(ValueError, match="positive"):
                create_embedding(v, d, backend)


def test_create_embedding_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown embedding storage"):
        create_embedding(8, 4, "disk")


# --- mmap: the file is the table ------------------------------------------

def test_a_newly_created_file_is_initialised():
    emb = MmapEmbedding(512, 8)
    w = emb.weight.detach()
    assert float(w.std()) > 0.005, "a fresh table should be N(0, 0.02), not zeros"
    assert emb.path.exists()
    assert emb.path.stat().st_size == 512 * 8 * 4


def test_an_existing_file_is_never_overwritten(tmp_path):
    """The bug: freshness was decided by sampling the first and last rows.

    A real checkpoint whose first and last embedding rows happened to be all
    zeros had its entire table replaced with N(0, 0.02) -- silently. Measured:
    the middle of the table went from std 1.105 to 0.019. Provenance is knowable
    (we created the file or we did not); content is not.
    """
    v, d = 64, 4
    data = torch.randn(v, d)
    data[0] = 0.0
    data[v - 1] = 0.0
    path = tmp_path / "e.dat"
    _write(path, v, d, data.numpy())
    emb = MmapEmbedding(v, d, path)
    assert torch.equal(emb.weight.detach(), data), "an existing table was clobbered"


def test_an_all_zero_existing_file_stays_zero(tmp_path):
    """Zero content is not evidence of freshness."""
    v, d = 16, 4
    path = tmp_path / "zeros.dat"
    path.write_bytes(b"\x00" * (v * d * 4))
    emb = MmapEmbedding(v, d, path)
    assert float(emb.weight.detach().abs().sum()) == 0.0


def test_a_file_of_the_wrong_size_is_refused(tmp_path):
    v, d = 8, 4
    path = tmp_path / "short.dat"
    path.write_bytes(b"\x00" * (v * d * 4 - 4))
    with pytest.raises(ValueError, match="expected"):
        MmapEmbedding(v, d, path)


def test_a_missing_file_is_created_with_its_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "e.dat"
    emb = MmapEmbedding(8, 4, path)
    assert path.exists() and path.stat().st_size == 8 * 4 * 4


def test_writes_go_through_to_the_file(tmp_path):
    v, d = 8, 4
    path = tmp_path / "e.dat"
    emb = MmapEmbedding(v, d, path)
    with torch.no_grad():
        emb.weight[2] = 5.0
    emb.flush()
    del emb
    mem = np.memmap(str(path), dtype=np.float32, mode="r", shape=(v, d))
    assert float(mem[2, 0]) == 5.0
    del mem


def test_gradients_flow_and_an_optimizer_step_lands_in_the_file(tmp_path):
    """The mapping is shared, so the existing optimizer path must work on it."""
    v, d = 16, 4
    path = tmp_path / "e.dat"
    emb = MmapEmbedding(v, d, path)
    opt = torch.optim.AdamW(emb.parameters(), lr=1e-2)
    before = emb.weight.detach().clone()
    loss = emb(torch.tensor([1, 2, 3])).sum()
    loss.backward()
    assert emb.weight.grad is not None
    opt.step()
    assert not torch.equal(before, emb.weight.detach())
    emb.flush()


def test_flush_is_safe_to_call_twice_and_on_a_dead_module(tmp_path):
    emb = MmapEmbedding(8, 4, tmp_path / "e.dat")
    emb.flush()
    emb.flush()
    orphan = MmapEmbedding.__new__(MmapEmbedding)
    orphan.flush()          # no _mem attribute yet
    # flush() swallows AttributeError from the un-built instance and
    # ValueError from an already-released mapping; both must still return.
    assert orphan.flush() is None
    assert emb.flush() is None


def test_to_device_leaves_the_mapping_on_cpu(tmp_path):
    """Documented: the table stays CPU fp32 so only looked-up rows cross."""
    emb = MmapEmbedding(8, 4, tmp_path / "e.dat")
    emb.to("cpu")
    emb.float()
    assert emb.weight.device.type == "cpu"
    assert emb.weight.dtype == torch.float32


def test_forward_moves_rows_for_a_foreign_device(tmp_path):
    emb = MmapEmbedding(8, 4, tmp_path / "e.dat")
    out = emb(torch.tensor([1, 2]))
    assert out.shape == (2, 4)
    assert out.device.type == "cpu"


# --- the RAM -> mmap copy ---------------------------------------------------

def test_copy_embedding_to_mmap_is_exact(tmp_path):
    v, d = 40, 7
    data = torch.randn(v, d)
    path = tmp_path / "e.dat"
    copy_embedding_to_mmap(data, path)
    mem = np.memmap(str(path), dtype=np.float32, mode="r", shape=(v, d))
    assert np.array_equal(np.asarray(mem), data.numpy())
    del mem


def test_copy_embedding_to_mmap_handles_a_stale_larger_file(tmp_path):
    v, d = 8, 4
    path = tmp_path / "e.dat"
    path.write_bytes(b"\x00" * (100 * d * 4))
    data = torch.randn(v, d)
    copy_embedding_to_mmap(data, path)
    assert path.stat().st_size == v * d * 4, "the stale tail was left behind"
    mem = np.memmap(str(path), dtype=np.float32, mode="r", shape=(v, d))
    assert np.array_equal(np.asarray(mem), data.numpy())
    del mem


def test_copy_embedding_to_mmap_upcasts_a_low_precision_table(tmp_path):
    """The file format is fp32, so a bf16 RAM table has to widen, not truncate."""
    data = torch.randn(16, 4).to(torch.bfloat16)
    path = tmp_path / "e.dat"
    copy_embedding_to_mmap(data, path)
    emb = MmapEmbedding(16, 4, path)
    assert emb.weight.dtype == torch.float32
    assert torch.allclose(emb.weight.detach(), data.float(), atol=1e-2)


@pytest.mark.parametrize("chunk", [1, 3, 16, 1024])
def test_copy_embedding_to_mmap_is_chunk_size_independent(tmp_path, chunk):
    v, d = 33, 5
    data = torch.randn(v, d)
    path = tmp_path / f"e{chunk}.dat"
    copy_embedding_to_mmap(data, path, chunk=chunk)
    mem = np.memmap(str(path), dtype=np.float32, mode="r", shape=(v, d))
    assert np.array_equal(np.asarray(mem), data.numpy())
    del mem


def test_copy_embedding_to_mmap_rejects_a_non_matrix(tmp_path):
    with pytest.raises(ValueError):
        copy_embedding_to_mmap(torch.randn(4), tmp_path / "e.dat")
