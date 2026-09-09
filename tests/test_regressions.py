import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import IGNORE_INDEX, _preprocess_conversation, iter_texts
from rl import SmaulRL
from rwkv_x_core import CausalSelfAttention, RWKVXConfig
from syntheticdata import gen_system_linear_equations
from tokenizer import read_texts


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab = {"User:": 2, "hello": 3, "System:": 4, "rules": 5, "Assistant:": 6, "answer": 7, "\n\n": 8}

    def encode(self, text):
        return [self.vocab[word] for word in text.replace("\n\n", " \n\n ").split()]


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
    import re
    for _ in range(1000):
        item = gen_system_linear_equations()
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


def test_sft_masks_non_assistant_messages_and_malformed_entries():
    result = _preprocess_conversation([
        {"from": "user", "value": "hello"},
        None,
        {"from": "system", "value": "rules"},
        {"from": "assistant", "value": "answer"},
        {"from": "broken"},
    ], FakeTokenizer(), 64, 0)
    labels = result["labels"].tolist()
    ids = result["input_ids"].tolist()
    assistant = FakeTokenizer().encode("Assistant: answer\n\n")
    start = ids.index(assistant[0])
    assert labels[start] == IGNORE_INDEX
    assert labels[start + 1:] == assistant[1:] + [IGNORE_INDEX] * (64 - start - len(assistant))
    assert all(x == IGNORE_INDEX for x in labels[:start])


def test_rl_sampling_logprob_uses_same_transformed_distribution():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    filtered = SmaulRL._filter_logits(logits, 0.5, 2, 1.0)
    expected = torch.log_softmax(filtered, -1)
    assert torch.isneginf(expected[2:]).all()
    assert torch.allclose(expected[:2].exp().sum(), torch.tensor(1.0))
    token, stored = SmaulRL._sample(logits, 0.5, 2, 1.0)
    assert token in (0, 1)
    assert abs(stored - expected[token].item()) < 1e-6


def test_pretrain_partial_batch_helper_updates_resume(monkeypatch):
    import train

    class Resume:
        global_step = 0
        total_tokens = 0
        file_path = None
        record_index = 0
        buffer_tokens = []

    monkeypatch.setattr(train, "_optimizer_step", lambda *args: torch.tensor(2.0))
    resume = Resume()
    loss = train._train_pretrain_batch(None, object(), object(), resume, torch.device("cpu"), None,
                                       [torch.ones(4, dtype=torch.long)], [torch.zeros(4, dtype=torch.long)],
                                       "data.txt", 3, [1, 2])
    assert loss.item() == 2.0
    assert resume.global_step == 1
    assert resume.total_tokens == 4
    assert resume.file_path == "data.txt"
    assert resume.record_index == 3
    assert resume.buffer_tokens == [1, 2]


if __name__ == "__main__":
    test_moba_cached_matches_full_for_multiple_chunks()
    print("regression tests passed")
