#!/usr/bin/env python3

import argparse
import contextlib
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

if "--cpu" in sys.argv:
    threads = str(os.environ.get("SMAUL_CPU_THREADS") or max(1, (os.cpu_count() or 2) // 2))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, threads)
    os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER", "1")

import torch
from torch.optim import Optimizer

from backend import backend_device, require_backend
from dataset import PretrainStream, SFTDataset, discover_files, iter_texts, load_tokenizer, tokenizer_vocab_size
from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE
from stream_data import stream_dataset
from tokenizer import ensure_tokenizer
from rqt import RQTLion, prepare_mixed_rqt, prepare_rqt

STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Ctrl-C] Stop requested. Finishing current step, then saving checkpoint.")


signal.signal(signal.SIGINT, _sigint_handler)


def _format_size(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def _print_model_size(model):
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {parameters:,} parameters | trainable={trainable:,} | FP32={_format_size(parameters * 4)}")
    return parameters


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr, (b1, b2), decay = group["lr"], group["betas"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                avg = state["exp_avg"]
                grad = param.grad.float()
                if decay:
                    param.mul_(1 - lr * decay)
                update = avg.mul(b1).add(grad, alpha=1 - b1).sign()
                param.add_(update, alpha=-lr)
                avg.lerp_(grad, 1 - b2)
        return loss


def _build_optimizer(args, model):
    if args.rqt or args.mixed_rqt:
        return RQTLion(model, lr=args.lr, weight_decay=args.weight_decay)
    return Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def set_router_only_training(model, router_only):
    if not model.cfg.is_moe:
        raise ValueError("set_router_only_training requires cfg.is_moe=True")
    gates = {id(p) for m in model.modules() if isinstance(m, RWKV_CMix_MoE) for p in m.gate.parameters()}
    count = 0
    for p in model.parameters():
        p.requires_grad_(id(p) in gates if router_only else True)
        count += p.numel() if p.requires_grad else 0
    return count


class ResumeState:
    def __init__(self):
        self.global_step = 0
        self.total_tokens = 0
        self.file_path: Optional[str] = None
        self.record_index = 0
        self.epoch = 0
        self.buffer_tokens = []

    @classmethod
    def load(cls, path):
        state = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                raise RuntimeError(f"could not load resume state: {path}") from exc
            state.__dict__.update(data)
        return state

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2))
        os.replace(tmp, path)


def _save_rng_state(path):
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def _load_rng_state(path):
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "torch" not in state:
        raise RuntimeError(f"missing torch RNG state: {path}")
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_optimizer_checkpoint(optimizer, resume, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    _save_rng_state(checkpoint_dir / "rng_state.pt")
    resume.save(checkpoint_dir / "resume_state.json")


def save_checkpoint(model, optimizer, resume, output_dir, checkpoint_dir, tokenizer_path, save_dtype="fp32", save_optimizer=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "_orig_mod", model)
    model.save_pretrained(output_dir, dtype=save_dtype, include_upstream=False)
    bundled = output_dir / "tokenizer.json"
    if tokenizer_path.resolve() != bundled.resolve():
        shutil.copy2(tokenizer_path, bundled)
    if save_optimizer:
        _save_optimizer_checkpoint(optimizer, resume, checkpoint_dir)
    print(f"[SAVE COMPLETE] {output_dir}")


def _autocast(args, device):
    if args.rqt or args.mixed_rqt:
        return contextlib.nullcontext()
    if device.type == "cuda":
        dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if args.precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _optimizer_step(args, model, optimizer, xb, yb, device):
    optimizer.zero_grad(set_to_none=True)
    with _autocast(args, device):
        _, loss, _ = model(xb, labels=yb)
    if not torch.isfinite(loss):
        print(f"[WARN] non-finite loss {loss.item()}, skipping step")
        return None
    loss.backward()
    if args.rqt or args.mixed_rqt:
        optimizer.clip_grad_norm(1.0)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    optimizer.step()
    return loss


def _remote_token_stream(name, tokenizer, ctx_len, resume):
    dataset = file_path = None
    record = 0
    if resume.file_path:
        try:
            dataset, file_path = resume.file_path.split("::", 1)
        except ValueError as exc:
            raise ValueError("invalid remote resume position") from exc
        record = resume.record_index
    buffer_tokens = list(resume.buffer_tokens)
    for text, position in stream_dataset(name, start_dataset=dataset, start_file=file_path, start_record=record, with_position=True):
        buffer_tokens.extend(tokenizer.encode(text) + [tokenizer.eos_token_id])
        while len(buffer_tokens) >= ctx_len + 1:
            chunk = buffer_tokens[:ctx_len + 1]
            del buffer_tokens[:ctx_len]
            yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:]), f"{position[0]}::{position[1]}", position[2], list(buffer_tokens)


def _train_batch(args, model, optimizer, resume, device, batch_x, batch_y, path, record, buffer_tokens):
    xb, yb = torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)
    loss = _optimizer_step(args, model, optimizer, xb, yb, device)
    resume.file_path, resume.record_index, resume.buffer_tokens = path, record, buffer_tokens
    if loss is None:
        return None
    resume.global_step += 1
    resume.total_tokens += xb.numel()
    return loss


def train_pretrain(args, model, optimizer, resume, device, tokenizer):
    if args.stream_dataset != "none":
        stream = _remote_token_stream(args.stream_dataset, tokenizer, args.ctx_len, resume)
        remote = True
    else:
        stream = PretrainStream(Path(args.dataset_dir), tokenizer, args.ctx_len, resume_file=resume.file_path, resume_record=resume.record_index, buffer_tokens=resume.buffer_tokens)
        remote = False
    model.train()
    batch_x, batch_y = [], []
    t0 = time.perf_counter()
    tokens_since_log = 0
    for item in stream:
        if remote:
            x, y, path, record, buffer_tokens = item
        else:
            x, y, position = item
            path, record, buffer_tokens = position[0], position[1], stream.buffer_tokens
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) < args.batch_size:
            continue
        loss = _train_batch(args, model, optimizer, resume, device, batch_x, batch_y, path, record, buffer_tokens)
        batch_x, batch_y = [], []
        if loss is None:
            continue
        tokens_since_log += args.ctx_len * args.batch_size
        if resume.global_step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {resume.global_step} | loss {loss.item():.4f} | {tokens_since_log / max(elapsed, 1e-9):.1f} tok/s | tokens {resume.total_tokens:,}")
            t0, tokens_since_log = time.perf_counter(), 0
        if resume.global_step % args.save_every == 0:
            save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path), args.save_dtype, False)
        if resume.global_step % args.optimizer_save_every == 0:
            _save_optimizer_checkpoint(optimizer, resume, Path(args.checkpoint_dir))
        if STOP_REQUESTED:
            break

    if batch_x and not STOP_REQUESTED:
        loss = _train_batch(args, model, optimizer, resume, device, batch_x, batch_y, path, record, buffer_tokens)
        batch_x, batch_y = [], []
        if loss is not None:
            resume.global_step += 1
            resume.total_tokens += len(batch_x) * args.ctx_len

def train_sft(args, model, optimizer, resume, device, tokenizer):
    dataset = SFTDataset(Path(args.dataset_dir), tokenizer, args.ctx_len)
    model.train()
    for epoch in range(resume.epoch, args.epochs):
        generator = torch.Generator().manual_seed(epoch)
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        start = resume.record_index if epoch == resume.epoch else 0
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=permutation[start:])
        consumed = start
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = _optimizer_step(args, model, optimizer, xb, yb, device)
            consumed += len(xb)
            resume.epoch, resume.record_index = epoch, consumed
            if loss is None:
                continue
            resume.global_step += 1
            resume.total_tokens += xb.numel()
            if resume.global_step % args.log_every == 0:
                print(f"epoch {epoch} step {resume.global_step} | loss {loss.item():.4f}")
            if resume.global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path), args.save_dtype, False)
            if resume.global_step % args.optimizer_save_every == 0:
                _save_optimizer_checkpoint(optimizer, resume, Path(args.checkpoint_dir))
            if STOP_REQUESTED:
                break
        if STOP_REQUESTED:
            break
        resume.epoch, resume.record_index = epoch + 1, 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--stream_dataset", choices=["none", "hindi", "english", "openthoughts", "all"], default="none")
    parser.add_argument("--output_dir", default="./SmaulNative")
    parser.add_argument("--checkpoint_dir", default="./SmaulNative")
    parser.add_argument("--tokenizer_path", default="./SmaulNative/tokenizer.json")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=65536)
    parser.add_argument("--tokenizer_max_records", type=int, default=5_000_000)
    parser.add_argument("--n_embd", type=int, default=832)
    parser.add_argument("--n_layer", type=int, default=17)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--n_moba_layer", type=int, default=3)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--save_dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--optimizer_save_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--rqt", action="store_true")
    parser.add_argument("--rqt_bits", type=int, choices=[4, 6, 8], default=6)
    parser.add_argument("--mixed_rqt", action="store_true")
    parser.add_argument("--router_only", action="store_true")
    return parser.parse_args()


def _tokenizer_path(args):
    return Path(args.tokenizer_path)


def _load_or_build_tokenizer(args):
    path = _tokenizer_path(args)
    if path.exists() and tokenizer_vocab_size(path) == args.tokenizer_vocab_size:
        return load_tokenizer(path)
    files = discover_files(Path(args.dataset_dir))
    return ensure_tokenizer(path, iter_texts(files, max_records=args.tokenizer_max_records), args.tokenizer_vocab_size, max_records=args.tokenizer_max_records)


def _checkpoint_config(output_dir):
    path = Path(output_dir) / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _build_model(args):
    expected = dict(vocab_size=args.tokenizer_vocab_size, n_embd=args.n_embd, n_layer=args.n_layer,
                    n_moba_layer=args.n_moba_layer, head_size=args.head_size, ctx_len_hint=args.ctx_len)
    checkpoint = _checkpoint_config(args.output_dir)
    if checkpoint and (Path(args.output_dir) / "model.safetensors").exists():
        keys = ("vocab_size", "n_embd", "n_layer", "n_moba_layer", "head_size", "ctx_len_hint")
        if all(checkpoint.get(key) == expected[key] for key in keys):
            return RWKVXModel.from_pretrained(args.output_dir)
    return RWKVXModel(expected)


def main():
    args = parse_args()
    if args.rqt or args.mixed_rqt:
        args.precision = "fp32"
    elif args.precision is None:
        args.precision = "fp16" if torch.cuda.is_available() and not args.cpu else "fp32"
    backend = require_backend(torch, force_cpu=args.cpu)
    device = backend_device(torch, backend)
    if device.type == "cpu":
        import cpu
        cpu.configure()
    print(f"[DEVICE] {device} | precision={args.precision} | rqt={args.rqt or args.mixed_rqt}")
    tokenizer = _load_or_build_tokenizer(args)
    model = _build_model(args).to(device)
    if args.mixed_rqt:
        prepare_mixed_rqt(model)
    elif args.rqt:
        prepare_rqt(model, args.rqt_bits)
    if args.router_only:
        trainable = set_router_only_training(model, True)
        print(f"[MOE] router-only trainable={trainable:,}")
    if args.compile:
        model = torch.compile(model)
    optimizer = _build_optimizer(args, model)
    resume = ResumeState.load(Path(args.checkpoint_dir) / "resume_state.json") if args.resume else ResumeState()
    if args.resume:
        optimizer_path = Path(args.checkpoint_dir) / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
        _load_rng_state(Path(args.checkpoint_dir) / "rng_state.pt")
    if args.mode == "pretrain":
        train_pretrain(args, model, optimizer, resume, device, tokenizer)
    else:
        train_sft(args, model, optimizer, resume, device, tokenizer)
    save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path), args.save_dtype)


if __name__ == "__main__":
    main()
