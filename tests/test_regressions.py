import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import iter_texts
from rwkv_x_core import CausalSelfAttention, RWKVXConfig
from syntheticdata import gen_system_linear_equations
from tokenizer import read_texts


def test_plain_text_resume_skips_completed_record(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("first\n\nsecond\n")
    files = [path.resolve()]
    assert [x[0] for x in iter_texts(files)] == ["first", "second"]
    assert [x[0] for x in iter_texts(files, str(path.resolve()), 1)] == ["second"]
    assert list(iter_texts(files, str(path.resolve()), 2)) == []


def test_tokenizer_input_order_is_stable(tmp_path):
    (tmp_path / "b.txt").write_text("beta")
    (tmp_path / "a.txt").write_text("alpha")
    assert list(read_texts(tmp_path)) == ["alpha", "beta"]


def test_generated_linear_system_is_nonsingular():
    for _ in range(1000):
        item = gen_system_linear_equations()
        lines = item["instruction"].splitlines()
        a1, b1 = map(int, lines[1].split("x + ")[0:2]) if False else (None, None)
        import re
        m = re.findall(r"(-?\d+)x \+ (-?\d+)y", item["instruction"])
        assert len(m) == 2
        a1, b1 = map(int, m[0])
        a2, b2 = map(int, m[1])
        assert a1 * b2 - a2 * b1 != 0


def test_moba_cached_matches_full_for_multiple_chunks():
    torch.manual_seed(0)
    cfg = RWKVXConfig(n_embd=64, head_size=16, moba_chunk_size=8, moba_topk=2, n_layer=1)
    att = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 37, 64)
    prompt = x[:, :29]
    step = x[:, 29:30]
    _, cache = att(prompt, use_cache=True)
    cached, _ = att(step, cache=cache, use_cache=True)
    full, _ = att(torch.cat((prompt, step), 1))
    assert torch.allclose(cached, full[:, -1:], rtol=1e-4, atol=1e-5)
