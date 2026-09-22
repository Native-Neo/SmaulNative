#!/usr/bin/env python3
"""SmaulLinear FP8 trainer: pretraining with tiled E4M3 weights, FP32 Lion, resume-free checkpoints."""
import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import torch

from dataset import PretrainStream, discover_files, iter_texts, load_tokenizer
from fp8_tile import fp8_modules
from smaul_linear import LinearConfig, SmaulLinear
from tokenizer import ensure_tokenizer

STOP = False
def _h(sig, fr):
    global STOP
    STOP = True
    print("\n[stop] finishing step then saving")
signal.signal(signal.SIGINT, _h)

class Lion:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), wd=0.01):
        self.p = [p for p in params if p.requires_grad]
        self.lr, self.b1, self.b2, self.wd = lr, betas[0], betas[1], wd
        self.m = {}
    def zero_grad(self, model=None):
        for p in self.p: p.grad = None
        if model is not None:
            for _, m in fp8_modules(model): m._gw = None
    @torch.no_grad()
    def _clip(self, mods, mx=1.0):
        gs = [m._gw for _, m in mods if m._gw is not None] + [p.grad.float() for p in self.p if p.grad is not None]
        if not gs: return 0.0
        t = torch.stack([g.pow(2).sum() for g in gs]).sum().sqrt()
        if t > mx:
            s = mx / (t + 1e-6)
            for _, m in mods:
                if m._gw is not None: m._gw.mul_(s)
            for p in self.p:
                if p.grad is not None: p.grad.mul_(s)
        return t.item()
    @torch.no_grad()
    def step(self, model):
        mods = fp8_modules(model)
        self._clip(mods)
        for _, m in mods:
            if m._gw is None: continue
            g = m._gw.float().contiguous()
            st = self.m.setdefault(m, torch.zeros_like(g))
            upd = (st * self.b1 + g * (1 - self.b1)).sign() * self.lr
            st.mul_(self.b2).add_(g, alpha=1 - self.b2)
            m.requant(upd, self.lr * self.wd)
        for p in self.p:
            if p.grad is None: continue
            g = p.grad.float()
            st = self.m.setdefault(p, torch.zeros_like(p, dtype=torch.float32))
            upd = st.mul(self.b1).add(g, alpha=1 - self.b1).sign()
            if self.wd: p.mul_(1 - self.lr * self.wd)
            p.add_(upd, alpha=-self.lr)
            st.mul_(self.b2).add_(g, alpha=1 - self.b2)
    def state_dict(self):
        return {"lr": self.lr, "wd": self.wd, "betas": [self.b1, self.b2]}
    def load_state_dict(self, d):
        self.lr = d.get("lr", self.lr); self.wd = d.get("wd", self.wd)

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def _tok(args):
    tp = Path(args.tokenizer)
    if tp.exists():
        from tokenizer import load as _load
        t = _load(tp)
        if t.get_vocab_size() == args.vocab and t.data.get("version") == 6: return t
        print("[tok] vocab mismatch, rebuilding")
    files = discover_files(Path(args.data))
    texts = (t for t, _, _ in iter_texts(files))
    return ensure_tokenizer(tp, texts, args.vocab)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="./datasets")
    a.add_argument("--out", default="./runs/linear")
    a.add_argument("--tokenizer", default="./runs/linear/tokenizer.json")
    a.add_argument("--vocab", type=int, default=8000)
    a.add_argument("--d", type=int, default=512)
    a.add_argument("--layers", type=int, default=8)
    a.add_argument("--heads", type=int, default=8)
    a.add_argument("--ctx", type=int, default=256)
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--steps", type=int, default=1000)
    a.add_argument("--lr", type=float, default=2e-4)
    a.add_argument("--wd", type=float, default=0.01)
    a.add_argument("--log_every", type=int, default=10)
    a.add_argument("--save_every", type=int, default=200)
    a.add_argument("--tok_records", type=int, default=200000)
    a.add_argument("--threads", type=int, default=2)
    args = a.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = _tok(args)
    tok.save(args.tokenizer)
    cfg = LinearConfig(vocab_size=args.vocab, d_model=args.d, n_layer=args.layers, n_heads=args.heads)
    model = SmaulLinear(cfg)
    opt = Lion(list(model.parameters()), lr=args.lr, wd=args.wd)
    wrap = load_tokenizer(args.tokenizer)
    stream = PretrainStream(Path(args.data), wrap, args.ctx)
    model.train()
    bx, by, step, toks, t0, since = [], [], 0, 0, time.perf_counter(), 0
    for x, y, _ in stream:
        if STOP or step >= args.steps: break
        bx.append(x); by.append(y)
        if len(bx) < args.batch: continue
        xb, yb = torch.stack(bx), torch.stack(by)
        bx, by = [], []
        opt.zero_grad(model)
        _, loss = model(xb, yb)
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss, skip step {step}");
            continue
        loss.backward()
        opt.step(model)
        step += 1
        toks += xb.numel(); since += xb.numel()
        if step % args.log_every == 0:
            el = time.perf_counter() - t0
            mem = sum(p.numel() * p.element_size() for p in model.parameters())
            mem += sum(b.numel() * b.element_size() for b in model.buffers())
            print(f"step {step} loss {loss.item():.4f} {since / max(el, 1e-9):.1f} tok/s stored {mem / 1048576:.1f}MiB")
            t0, since = time.perf_counter(), 0
        if step % args.save_every == 0:
            model.save_pretrained(out)
            torch.save(opt.state_dict(), out / "optimizer.pt")
            print(f"[save] {out}")
    model.save_pretrained(out)
    torch.save(opt.state_dict(), out / "optimizer.pt")
    print(f"[done] steps={step} tokens={toks}")

if __name__ == "__main__":
    main()
