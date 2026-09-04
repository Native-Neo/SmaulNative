#!/usr/bin/env python3
import os
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_cpu_dir = str(Path(__file__).resolve().parent)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)

_default_threads = str(os.environ.get("SMAUL_CPU_THREADS") or max(1, (os.cpu_count() or 2) // 2))
os.environ.setdefault("OMP_NUM_THREADS", _default_threads)
os.environ.setdefault("MKL_NUM_THREADS", _default_threads)

# The optimized CPU entrypoint enables torch.compile by default. Use --no-compile
# when debugging or when the compile startup cost is undesirable for a short run.
if "--no-compile" in sys.argv:
    sys.argv.remove("--no-compile")
else:
    if "--compile" not in sys.argv:
        sys.argv.append("--compile")

import torch
import train
try:
    from cpu_backend import NativeLion, configure
except ImportError:
    from cpu.cpu_backend import NativeLion, configure

threads = configure()
train.Lion = NativeLion

_old_pretrain = train.train_pretrain


def train_pretrain(args, model, optimizer, resume, device, tokenizer):
    if resume.file_path is not None and not train.Path(resume.file_path).is_file():
        raise FileNotFoundError(f"resume dataset file no longer exists: {resume.file_path}")
    stream = train.PretrainStream(train.Path(args.dataset_dir), tokenizer, args.ctx_len,
                                  resume_file=resume.file_path, resume_record=resume.record_index,
                                  buffer_tokens=resume.buffer_tokens)
    model.train()
    batch_x, batch_y = [], []
    t0 = train.time.perf_counter()
    tok_since = 0
    for x, y, pos in stream:
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) < args.batch_size:
            continue
        if args.batch_size == 1:
            xb, yb = batch_x[0].unsqueeze(0), batch_y[0].unsqueeze(0)
        else:
            xb, yb = torch.stack(batch_x), torch.stack(batch_y)
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(xb, labels=yb)
        if not torch.isfinite(loss):
            print(f"[WARN] non-finite loss {loss.item()}, skipping step")
            batch_x, batch_y = [], []
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        resume.global_step += 1
        resume.total_tokens += xb.numel()
        resume.file_path, resume.record_index = pos
        resume.buffer_tokens = list(stream.buffer_tokens)
        tok_since += xb.numel()
        if resume.global_step % args.log_every == 0:
            dt = train.time.perf_counter() - t0
            print(f"step {resume.global_step} | loss {loss.item():.4f} | {tok_since/max(dt,1e-9):.1f} tok/s | tokens {resume.total_tokens:,} | cpu_threads {threads}")
            t0 = train.time.perf_counter()
            tok_since = 0
        batch_x, batch_y = [], []
        if resume.global_step % args.save_every == 0:
            train.save_checkpoint(model, optimizer, resume, train.Path(args.output_dir), train.Path(args.checkpoint_dir), train.Path(args.tokenizer_path))
        if train.STOP_REQUESTED:
            break


train.train_pretrain = train_pretrain

if __name__ == "__main__":
    train.main()
