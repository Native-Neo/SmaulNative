#!/usr/bin/env python3
# train.py -- single entrypoint for RWKV-X pretrain + SFT.

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer

from rwkv_x_core import RWKVXModel, config_for_target_params
from router_utils import set_router_only_training
from dataset import load_tokenizer, tokenizer_vocab_size, PretrainStream, SFTDataset, iter_texts, discover_files
from tokenizer import train_tokenizer
import qat

DEFAULT_TARGET_PARAMS = 256_000_000
STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Ctrl-C] Stop requested. Finishing current step, then saving checkpoint.")


signal.signal(signal.SIGINT, _sigint_handler)


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), wd = group["lr"], group["betas"], group["weight_decay"]
            params, grads, avgs = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                avgs.append(state["exp_avg"])
            if not params:
                continue
            if wd:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            updates = torch._foreach_mul(avgs, beta1)
            torch._foreach_add_(updates, grads, alpha=1.0 - beta1)
            torch._foreach_add_(params, torch._foreach_sign(updates), alpha=-lr)
            torch._foreach_lerp_(avgs, grads, 1.0 - beta2)
        return loss


class ResumeState:
    def __init__(self):
        self.global_step = 0
        self.total_tokens = 0
        self.file_path: Optional[str] = None
        self.record_index = 0
        self.epoch = 0
        self.buffer_tokens = []

    @classmethod
    def load(cls, path: Path):
        s = cls()
        if path.exists():
            try:
                d = json.loads(path.read_text())
                s.global_step = d.get("global_step", 0)
                s.total_tokens = d.get("total_tokens", 0)
                s.file_path = d.get("file_path")
                s.record_index = d.get("record_index", 0)
                s.epoch = d.get("epoch", 0)
                s.buffer_tokens = d.get("buffer_tokens", [])
            except Exception as e:
                print(f"[WARN] could not load resume state: {e}")
        return s

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "global_step": self.global_step,
            "total_tokens": self.total_tokens,
            "file_path": self.file_path,
            "record_index": self.record_index,
            "epoch": self.epoch,
            "buffer_tokens": self.buffer_tokens,
        }, indent=2))
        os.replace(tmp, path)


def _save_rng_state(path: Path):
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_rng_state(path: Path):
    if not path.exists():
        return
    try:
        state = torch.load(path, map_location="cpu")
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
        print("[RESUME] restored RNG state")
    except Exception as e:
        print(f"[WARN] could not restore RNG state: {e}")


def save_checkpoint(model: RWKVXModel, optimizer: Optimizer, resume: ResumeState,
                    output_dir: Path, checkpoint_dir: Path, tokenizer_path: Path):
    print("\n[SAVE] Saving checkpoint...")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    bundled_tok = output_dir / "tokenizer.json"
    if tokenizer_path.resolve() != bundled_tok.resolve():
        shutil.copy2(tokenizer_path, bundled_tok)
    tmp = checkpoint_dir / "optimizer.pt.tmp"
    torch.save(optimizer.state_dict(), tmp)
    os.replace(tmp, checkpoint_dir / "optimizer.pt")
    resume.save(checkpoint_dir / "resume_state.json")
    _save_rng_state(checkpoint_dir / "rng_state.pt")
    print(f"[SAVE COMPLETE] model -> {output_dir}, optimizer/resume -> {checkpoint_dir}\n")


def train_pretrain(args, model, optimizer, resume, device, tokenizer):
    if resume.file_path is not None and not Path(resume.file_path).is_file():
        raise FileNotFoundError(f"resume dataset file no longer exists: {resume.file_path}")
    stream = PretrainStream(Path(args.dataset_dir), tokenizer, args.ctx_len,
                             resume_file=resume.file_path, resume_record=resume.record_index,
                             buffer_tokens=resume.buffer_tokens)
    model.train()
    batch_x, batch_y = [], []
    t0 = time.perf_counter()
    tok_since = 0
    for x, y, pos in stream:
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) < args.batch_size:
            continue
        xb = torch.stack(batch_x).to(device)
        yb = torch.stack(batch_y).to(device)
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
            dt = time.perf_counter() - t0
            print(f"step {resume.global_step} | loss {loss.item():.4f} | {tok_since/max(dt,1e-9):.1f} tok/s | tokens {resume.total_tokens:,}")
            t0 = time.perf_counter()
            tok_since = 0
        batch_x, batch_y = [], []
        if resume.global_step % args.save_every == 0:
            save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path))
        if STOP_REQUESTED:
            break


def train_sft(args, model, optimizer, resume, device, tokenizer):
    dataset = SFTDataset(Path(args.dataset_dir), tokenizer, args.ctx_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model.train()
    for epoch in range(resume.epoch, args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(xb, labels=yb)
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss {loss.item()}, skipping step")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            resume.global_step += 1
            resume.total_tokens += xb.numel()
            if resume.global_step % args.log_every == 0:
                print(f"epoch {epoch} step {resume.global_step} | loss {loss.item():.4f}")
            if resume.global_step % args.save_every == 0:
                resume.epoch = epoch
                save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path))
            if STOP_REQUESTED:
                break
        if STOP_REQUESTED:
            break
        resume.epoch = epoch + 1


def parse_args():
    p = argparse.ArgumentParser(description="RWKV-X trainer (pretrain + sft, CPU-safe)")
    p.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    p.add_argument("--dataset_dir", type=str, default="./datasets")
    p.add_argument("--output_dir", type=str, default="./SmaulNative")
    p.add_argument("--checkpoint_dir", type=str, default="./SmaulNative")
    p.add_argument("--tokenizer_path", type=str, default="./SmaulNative/tokenizer.json")
    p.add_argument("--tokenizer_vocab_size", type=int, default=65536)
    p.add_argument("--target_params", type=int, default=DEFAULT_TARGET_PARAMS)
    p.add_argument("--n_embd", type=int, default=832)
    p.add_argument("--head_size", type=int, default=64)
    p.add_argument("--n_moba_layer", type=int, default=5)
    p.add_argument("--ctx_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["lion", "adamw"], default="lion")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--new_data", action="store_true")
    p.add_argument("--train_router_only", action="store_true")
    p.add_argument("--qat", action="store_true")
    p.add_argument("--qat_calib_batches", type=int, default=64)
    p.add_argument("--qat_export_dir", type=str, default=None)
    args = p.parse_args()
    if args.target_params <= 0:
        p.error("--target_params must be > 0")
    if args.n_embd <= 0 or args.head_size <= 0:
        p.error("--n_embd and --head_size must be > 0")
    if args.n_embd % args.head_size:
        p.error(f"--n_embd ({args.n_embd}) must be divisible by --head_size ({args.head_size})")
    return args


def build_model(args, tokenizer) -> RWKVXModel:
    output_dir = Path(args.output_dir)
    if (output_dir / "config.json").exists() and (output_dir / "model.safetensors").exists():
        print(f"[RESUME] loading model from {output_dir}")
        return RWKVXModel.from_pretrained(output_dir)
    print("[INIT] creating new model")
    cfg = config_for_target_params(args.target_params, vocab_size=tokenizer_vocab_size(tokenizer),
                                    n_embd=args.n_embd, n_moba_layer=args.n_moba_layer,
                                    head_size=args.head_size)
    actual_params = cfg.approx_param_count()
    if abs(actual_params - args.target_params) / args.target_params > 0.10:
        raise ValueError(f"requested {args.target_params:,} params, closest supported architecture is {actual_params:,} ({abs(actual_params - args.target_params) / args.target_params:.1%} away)")
    cfg.ctx_len_hint = args.ctx_len
    model = RWKVXModel(cfg)
    print(f"[MODEL] n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_moba_layer={cfg.n_moba_layer} vocab_size={cfg.vocab_size} -> {model.num_parameters()/1e6:.1f}M params")
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    tokenizer_path = Path(args.tokenizer_path)
    output_dir = Path(args.output_dir)
    bundled_tok = output_dir / "tokenizer.json"
    if (output_dir / "config.json").exists() and bundled_tok.exists() and bundled_tok.resolve() != tokenizer_path.resolve():
        print(f"[RESUME] using tokenizer bundled with existing checkpoint: {bundled_tok}")
        tokenizer_path = bundled_tok
    if not tokenizer_path.exists():
        print(f"[TOKENIZER] {tokenizer_path} not found -- training one on {args.dataset_dir} (vocab_size={args.tokenizer_vocab_size})")
        train_tokenizer(Path(args.dataset_dir), tokenizer_path, args.tokenizer_vocab_size)
    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model(args, tokenizer).to(device)
    if args.train_router_only:
        n_trainable = set_router_only_training(model, True)
        n_total = model.num_parameters()
        print(f"[ROUTER-ONLY] frozen everything except router gates: {n_trainable:,} / {n_total:,} params trainable ({n_trainable / n_total * 100:.2f}%)")
    if args.qat:
        n = qat.prepare_qat(model)
        print(f"[QAT] fake-quantizing {n} Channel-Mix linear(s); calibrating on {args.qat_calib_batches} batches from {args.dataset_dir} ...")
        calib_files = discover_files(Path(args.dataset_dir))
        calib_texts = (text for text, _path, _idx in iter_texts(calib_files))
        done = qat.calibrate(model, tokenizer, calib_texts, args.ctx_len, device, max_batches=args.qat_calib_batches)
        print(f"[QAT] calibrated on {done} batches")
    opt_cls = Lion if args.optimizer == "lion" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=args.learning_rate)
    checkpoint_dir = Path(args.checkpoint_dir)
    opt_path = checkpoint_dir / "optimizer.pt"
    if opt_path.exists() and not args.new_data:
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print("[RESUME] loaded optimizer state")
        except Exception as e:
            print(f"[WARN] could not restore optimizer: {e}")
    resume = ResumeState.load(checkpoint_dir / "resume_state.json")
    if args.new_data:
        resume = ResumeState()
    else:
        _load_rng_state(checkpoint_dir / "rng_state.pt")
    print(f"[RESUME] step={resume.global_step:,} tokens={resume.total_tokens:,} file={resume.file_path} record={resume.record_index} epoch={resume.epoch}")
    try:
        if args.mode == "pretrain":
            train_pretrain(args, model, optimizer, resume, device, tokenizer)
        else:
            train_sft(args, model, optimizer, resume, device, tokenizer)
    finally:
        save_checkpoint(model, optimizer, resume, Path(args.output_dir), checkpoint_dir, tokenizer_path)
    if args.qat and args.qat_export_dir:
        import copy
        print(f"[QAT] converting to real packed int3 weights -> {args.qat_export_dir}")
        exported = copy.deepcopy(model).cpu()
        n = qat.convert_qat(exported)
        exported.save_pretrained(Path(args.qat_export_dir))
        shutil.copy2(tokenizer_path, Path(args.qat_export_dir) / "tokenizer.json")
        print(f"[QAT] converted {n} linear(s), exported to {args.qat_export_dir}")


if __name__ == "__main__":
    main()
