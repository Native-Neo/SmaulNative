import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import train


class DummyModel:
    def train(self):
        return self


class DummyOptimizer:
    def state_dict(self):
        return {}


def test_final_partial_batch_is_flushed(monkeypatch):
    class Args:
        stream_dataset = "none"
        dataset_dir = "."
        ctx_len = 4
        batch_size = 2
        log_every = 999
        save_every = 999
        optimizer_save_every = 999
        output_dir = "."
        checkpoint_dir = "."
        tokenizer_path = "tokenizer.json"
        save_dtype = "fp32"

    class Stream:
        buffer_tokens = []
        def __iter__(self):
            yield torch.arange(4), torch.arange(1, 5), ("data.txt", 1)

    calls = []
    monkeypatch.setattr(train, "PretrainStream", lambda *args, **kwargs: Stream())
    monkeypatch.setattr(train, "_train_pretrain_batch", lambda *args: calls.append(len(args[6])) or torch.tensor(1.0))
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: None)
    resume = train.ResumeState()
    train.STOP_REQUESTED = False
    train.train_pretrain(Args(), DummyModel(), DummyOptimizer(), resume, torch.device("cpu"), object(), None)
    assert calls == [1]


def test_checkpoint_waits_for_optimizer_checkpoint(monkeypatch):
    class Args:
        stream_dataset = "none"
        dataset_dir = "."
        ctx_len = 4
        batch_size = 1
        log_every = 999
        save_every = 1
        optimizer_save_every = 2
        output_dir = "."
        checkpoint_dir = "."
        tokenizer_path = "tokenizer.json"
        save_dtype = "fp32"

    class Stream:
        buffer_tokens = []
        def __iter__(self):
            yield torch.arange(4), torch.arange(1, 5), ("data.txt", 1)
            yield torch.arange(4), torch.arange(1, 5), ("data.txt", 2)

    saves = []

    def train_batch(*args):
        args[3].global_step += 1
        return torch.tensor(1.0)

    monkeypatch.setattr(train, "PretrainStream", lambda *args, **kwargs: Stream())
    monkeypatch.setattr(train, "_train_pretrain_batch", train_batch)
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: saves.append(args[2].global_step))
    resume = train.ResumeState()
    train.STOP_REQUESTED = False
    train.train_pretrain(Args(), DummyModel(), DummyOptimizer(), resume, torch.device("cpu"), object(), None)
    assert saves == [1, 2]


def test_model_save_does_not_write_resume_state(tmp_path):
    class Model:
        def save_pretrained(self, output_dir, dtype="fp32", include_upstream=False):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    resume = train.ResumeState()
    train.save_checkpoint(Model(), DummyOptimizer(), resume, tmp_path / "model", tmp_path / "checkpoint", tokenizer, save_optimizer=False)
    assert not (tmp_path / "checkpoint" / "resume_state.json").exists()


def test_resume_state_round_trip(tmp_path):
    state = train.ResumeState()
    state.global_step = 7
    state.total_tokens = 123
    state.file_path = "data.txt"
    state.record_index = 4
    state.buffer_tokens = [1, 2]
    path = tmp_path / "resume.json"
    state.save(path)
    loaded = train.ResumeState.load(path)
    assert loaded.__dict__ == state.__dict__


def test_corrupt_resume_state_fails_loudly(tmp_path):
    path = tmp_path / "resume_state.json"
    path.write_text("not json")
    with pytest.raises(RuntimeError, match="could not load resume state"):
        train.ResumeState.load(path)


def test_invalid_rng_checkpoint_fails_loudly(tmp_path):
    path = tmp_path / "rng_state.pt"
    torch.save({}, path)
    with pytest.raises(RuntimeError, match="missing torch RNG state"):
        train._load_rng_state(path)


def test_malformed_remote_resume_position_fails_loudly():
    resume = train.ResumeState()
    resume.file_path = "local_dataset.txt"
    with pytest.raises(ValueError, match="invalid remote resume position"):
        next(train._remote_token_stream("hindi", object(), 4, resume))
