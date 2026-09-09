import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import train


def test_final_partial_batch_is_flushed(monkeypatch):
    class Args:
        stream_dataset = "none"
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
    train.train_pretrain(Args(), object(), object(), resume, torch.device("cpu"), object(), None)
    assert calls == [1]


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
