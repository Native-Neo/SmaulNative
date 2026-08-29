#!/usr/bin/env python3
# train.py -- single entrypoint for RWKV-X pretrain + SFT, plain torch loop (no Lightning/DeepSpeed),
# Ctrl-C checkpointing, full resume, HF export. Training ctx_len is still a finite BPTT window --
# "unlimited context" is RWKV's inference-time state property, not free at train time.

import argparse
import json
import os
import shutil
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from rwkv_x_core import RWKVXConfig, RWKVXModel, config_for_target_params
from data import load_tokenizer, tokenizer_vocab_size, PretrainStream, SFTDataset
from tokenizer import train_tokenizer

STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Ctrl-C] Stop requested. Finishing current step, then saving checkpoint.")


signal.signal(signal.SIGINT, _sigint_handler)


# Lion optimizer (lighter on CPU RAM than Adam's two momentum buffers)

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


# Resume state

class ResumeState:
    def __init__(self):
        self.global_step = 0
        self.total_tokens = 0
        self.file_path: Optional[str] = None  # pretrain: last-consumed file
        self.record_index = 0                 # pretrain: last-consumed record within that file
        self.epoch = 0                        # sft: next epoch to run

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
        }, indent=2))
        os.replace(tmp, path)


# Checkpointing

def save_checkpoint(model: RWKVXModel, optimizer: Optimizer, resume: ResumeState,
                     output_dir: Path, checkpoint_dir: Path, tokenizer_path: Path):
    print("\n[SAVE] Saving checkpoint...")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)  # writes config.json, model.safetensors, upstream .pth

    # bundle the exact tokenizer this checkpoint was trained with, so later steps (SFT resume,
    # merge_moe.py, inference) never have to guess which tokenizer.json goes with which model.
    bundled_tok = output_dir / "tokenizer.json"
    if tokenizer_path.resolve() != bundled_tok.resolve():
        shutil.copy2(tokenizer_path, bundled_tok)

    tmp = checkpoint_dir / "optimizer.pt.tmp"
    torch.save(optimizer.state_dict(), tmp)
    os.replace(tmp, checkpoint_dir / "optimizer.pt")

    resume.save(checkpoint_dir / "resume_state.json")
    print(f"[SAVE COMPLETE] model -> {output_dir}, optimizer/resume -> {checkpoint_dir}\n")


# Training loops

def train_pretrain(args, model, optimizer, resume, device, tokenizer):
    stream = PretrainStream(Path(args.dataset_dir), tokenizer, args.ctx_len,
                             resume_file=resume.file_path, resume_record=resume.record_index)
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
        logits, loss, _ = model(xb, labels=yb)

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
        tok_since += xb.numel()

        if resume.global_step % args.log_every == 0:
            dt = time.perf_counter() - t0
            print(f"step {resume.global_step} | loss {loss.item():.4f} | "
                  f"{tok_since/max(dt,1e-9):.1f} tok/s | tokens {resume.total_tokens:,}")
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
    start_epoch = resume.epoch
    for epoch in range(start_epoch, args.epochs):
        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss, _ = model(xb, labels=yb)
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
        resume.epoch = epoch + 1
        if STOP_REQUESTED:
            break


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="RWKV-X godfather trainer (pretrain + sft, CPU-safe)")
    p.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    p.add_argument("--dataset_dir", type=str, default="./datasets")
    p.add_argument("--output_dir", type=str, default="./RWKV-X-256M")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--tokenizer_path", type=str, default="./tokenizer.json",
                    help="merged BPE tokenizer.json (see tokenizer.py); auto-trained on "
                         "--dataset_dir if missing")
    p.add_argument("--tokenizer_vocab_size", type=int, default=32768,
                    help="only used when auto-training a tokenizer.json that doesn't exist yet")

    p.add_argument("--target_params", type=int, default=256_000_000)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--head_size", type=int, default=64, help="n_embd must be divisible by this")
    p.add_argument("--n_moba_layer", type=int, default=3, help="MOBA sparse-attn blocks; 0 = pure RWKV-7")
    p.add_argument("--ctx_len", type=int, default=512, help="train-time BPTT window (CPU: keep this small)")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3, help="sft only")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["lion", "adamw"], default="lion")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--new_data", action="store_true",
                    help="reset dataset position, keep model/optimizer weights")
    return p.parse_args()


def build_model(args, tokenizer) -> RWKVXModel:
    output_dir = Path(args.output_dir)
    if (output_dir / "config.json").exists() and (output_dir / "model.safetensors").exists():
        print(f"[RESUME] loading model from {output_dir}")
        return RWKVXModel.from_pretrained(output_dir)

    print("[INIT] creating new model")
    vocab_size = tokenizer_vocab_size(tokenizer)
    cfg = config_for_target_params(args.target_params, vocab_size=vocab_size,
                                    n_embd=args.n_embd, n_moba_layer=args.n_moba_layer,
                                    head_size=args.head_size)
    cfg.ctx_len_hint = args.ctx_len
    model = RWKVXModel(cfg)
    print(f"[MODEL] n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_moba_layer={cfg.n_moba_layer} "
          f"vocab_size={cfg.vocab_size} -> {model.num_parameters()/1e6:.1f}M params")
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cpu":
        print("[NOTE] CPU-only: WKV recurrence runs as a plain Python loop (no compiled kernel exists "
              "for CPU). This will be slow, especially at larger ctx_len. Keep --ctx_len modest.")

    tokenizer_path = Path(args.tokenizer_path)
    output_dir = Path(args.output_dir)
    bundled_tok = output_dir / "tokenizer.json"
    if (output_dir / "config.json").exists() and bundled_tok.exists() and bundled_tok.resolve() != tokenizer_path.resolve():
        print(f"[RESUME] using tokenizer bundled with existing checkpoint: {bundled_tok}")
        tokenizer_path = bundled_tok
    if not tokenizer_path.exists():
        print(f"[TOKENIZER] {tokenizer_path} not found -- training one on {args.dataset_dir} "
              f"(vocab_size={args.tokenizer_vocab_size})")
        train_tokenizer(Path(args.dataset_dir), tokenizer_path, args.tokenizer_vocab_size)
    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model(args, tokenizer).to(device)

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
    print(f"[RESUME] step={resume.global_step:,} tokens={resume.total_tokens:,} "
          f"file={resume.file_path} record={resume.record_index} epoch={resume.epoch}")

    try:
        if args.mode == "pretrain":
            train_pretrain(args, model, optimizer, resume, device, tokenizer)
        else:
            train_sft(args, model, optimizer, resume, device, tokenizer)
    finally:
        save_checkpoint(model, optimizer, resume, Path(args.output_dir), checkpoint_dir, tokenizer_path)


if __name__ == "__main__":
    main()
