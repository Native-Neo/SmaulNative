#!/usr/bin/env python3
"""SmaulLinear FP8 trainer: pretraining with tiled E4M3 weights, FP32 Lion, resume-free checkpoints."""
import argparse
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

# Named model-size presets: preset name -> dict(vocab, d, layers, heads, ffn_mult).
# Effective fp32-equivalent params ~= 2*vocab*d + (5*layers+2)*d
#   + layers*(4*d*d + 3*d*int(d*ffn_mult)). FP8 per-tile scales add ~1-2% on top.
# Presets are added one per commit, largest first.
PRESETS: dict = {
    # ~1,040M params (1.024B target).
    "1B": {"vocab": 8000, "d": 2048, "layers": 24, "heads": 16, "ffn_mult": 2.0},
    # ~497M params (512M target).
    "512M": {"vocab": 8000, "d": 1536, "layers": 20, "heads": 12, "ffn_mult": 2.0},
    # ~250M params (256M target).
    "256M": {"vocab": 8000, "d": 1280, "layers": 14, "heads": 10, "ffn_mult": 2.0},
    # ~132M params (128M target).
    "128M": {"vocab": 8000, "d": 1024, "layers": 11, "heads": 8, "ffn_mult": 2.0},
    # ~65M params (64M target).
    "64M": {"vocab": 8000, "d": 768, "layers": 9, "heads": 12, "ffn_mult": 2.0},
    # ~32M params (32M target, matches previous defaults).
    "32M": {"vocab": 8000, "d": 512, "layers": 8, "heads": 8, "ffn_mult": 2.5},
    # ~16M params (16M target).
    "16M": {"vocab": 4000, "d": 512, "layers": 4, "heads": 8, "ffn_mult": 2.5},
    # ~7.8M params (8M target).
    "8M": {"vocab": 2000, "d": 448, "layers": 3, "heads": 7, "ffn_mult": 2.0},
    # ~4.2M params (4M target).
    "4M": {"vocab": 512, "d": 256, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~2.1M params (2M target).
    "2M": {"vocab": 256, "d": 256, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~1.05M params (1M target).
    "1M": {"vocab": 256, "d": 128, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~526K params (512K target).
    "512K": {"vocab": 128, "d": 128, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~256K params (256K target).
    "256K": {"vocab": 64, "d": 64, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~132K params (128K target).
    "128K": {"vocab": 64, "d": 64, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~67K params (64K target).
    "64K": {"vocab": 32, "d": 56, "layers": 2, "heads": 4, "ffn_mult": 2.0},
    # ~33K params (32K target).
    "32K": {"vocab": 32, "d": 32, "layers": 3, "heads": 2, "ffn_mult": 2.0},
}


def list_presets() -> dict:
    return dict(PRESETS)


def estimate_params(vocab: int, d: int, layers: int, ffn_mult: float) -> int:
    h = int(d * ffn_mult)
    return 2 * vocab * d + (5 * layers + 2) * d + layers * (4 * d * d + 3 * d * h)


def apply_preset(args) -> None:
    name = getattr(args, "preset", None)
    if not name:
        return
    try:
        p = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown --preset {name!r}; use --list-presets (have {sorted(PRESETS)})") from None
    for k in ("vocab", "d", "layers", "heads", "ffn_mult"):
        if k in p:
            setattr(args, k, p[k])

STOP = False
def _h(sig, fr):
    global STOP
    STOP = True
    print("\n[stop] finishing step then saving")

def install_handlers() -> None:
    # Install SIGINT/SIGTERM handlers explicitly from main() only.
    # Importing train (e.g. cpu/benchmark_full.py imports Lion) must not
    # hijack process signals as a side effect.
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _h)
        except (OSError, ValueError):
            pass

class Lion:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), wd=0.01, clip=1.0):
        self.p = [p for p in params if p.requires_grad]
        self.lr, self.b1, self.b2, self.wd = lr, betas[0], betas[1], wd
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
        self.clip = clip
        self.m = {}
    def zero_grad(self, model=None):
        for p in self.p:
            # Detach first so stale graph refs cannot accumulate.
            p.grad = None
        if model is not None:
            for _, m in fp8_modules(model):
                m._gw = None
        else:
            # Without the model we cannot clear tiled FP8 grads; warn once
            # because the next step would otherwise double-count them.
            import warnings
            warnings.warn("Lion.zero_grad called without model; FP8 _gw not cleared", RuntimeWarning,
                          stacklevel=2)
    @torch.no_grad()
    def _clip(self, mods, mx=1.0):
        gs = [m._gw for _, m in mods if m._gw is not None] + [p.grad.float() for p in self.p if p.grad is not None]
        if not gs:
            return 0.0
        # Filter non-finite grads: NaN never satisfies `t > mx`, so check first.
        for g in gs:
            if not bool(torch.isfinite(g).all()):
                return float("inf")
        # Accumulate in float64 to avoid fp32 overflow on large models.
        t = torch.stack([g.double().pow(2).sum() for g in gs]).sum().sqrt()
        if t > mx:
            s = mx / (t + 1e-6)
            for _, m in mods:
                if m._gw is not None:
                    m._gw.mul_(s)
            for p in self.p:
                if p.grad is not None:
                    p.grad.mul_(s)
        return float(t.item())
    @torch.no_grad()
    def step(self, model):
        mods = fp8_modules(model)
        norm = self._clip(mods, self.clip)
        if norm == float("inf"):
            # Non-finite grads would poison quantized weights via sign().
            # Clear them and skip the update; caller also guards loss.
            self.zero_grad(model)
            return norm
        live = set()
        for _, m in mods:
            if m._gw is None:
                continue
            g = m._gw.float().contiguous()
            st = self.m.get(m)
            if st is None or st.shape != g.shape:
                st = torch.zeros_like(g)
                self.m[m] = st
            elif st.device != g.device:
                st = st.to(g.device)
                self.m[m] = st
            live.add(m)
            m.fused_lion_requant(g, st, self.lr, self.wd, self.b1, self.b2)
        for p in self.p:
            if p.grad is None:
                continue
            g = p.grad.float()
            st = self.m.get(p)
            if st is None or st.shape != tuple(p.shape):
                st = torch.zeros_like(p, dtype=torch.float32)
                self.m[p] = st
            elif st.device != g.device:
                st = st.to(g.device)
                self.m[p] = st
            live.add(p)
            upd = st.mul(self.b1).add(g, alpha=1 - self.b1).sign()
            if self.wd:
                p.mul_(1 - self.lr * self.wd)
            p.add_(upd, alpha=-self.lr)
            st.mul_(self.b2).add_(g, alpha=1 - self.b2)
        # Evict momentum for dead params/modules (e.g. architecture change).
        for k in list(self.m):
            if k not in live:
                del self.m[k]
        return norm
    def state_dict(self):
        # Resume-free: momentum (self.m) is intentionally not saved.
        return {"lr": self.lr, "wd": self.wd, "betas": [self.b1, self.b2], "clip": self.clip}
    def load_state_dict(self, d):
        self.lr = d.get("lr", self.lr)
        self.wd = d.get("wd", self.wd)
        betas = d.get("betas", [self.b1, self.b2])
        try:
            self.b1, self.b2 = float(betas[0]), float(betas[1])
        except (TypeError, IndexError, ValueError):
            pass
        try:
            clip = float(d.get("clip", self.clip))
            if clip > 0:
                self.clip = clip
        except (TypeError, ValueError):
            pass

def _save_optimizer(out: Path, opt: "Lion") -> None:
    # JSON, not pickle: torch.save would be arbitrary-code-exec on load.
    (out / "optimizer.json").write_text(json.dumps(opt.state_dict(), indent=2), encoding="utf-8")


def _validate_args(args) -> None:
    for name in ("batch", "ctx", "steps", "d", "layers", "heads", "vocab", "threads", "log_every",
                 "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive, got {getattr(args, name)}")
    if args.d % args.heads != 0:
        raise ValueError(f"--d ({args.d}) must be divisible by --heads ({args.heads})")
    if not 0 < args.lr < 10:
        raise ValueError(f"--lr looks invalid: {args.lr}")
    if not 0 <= args.wd < 10:
        raise ValueError(f"--wd looks invalid: {args.wd}")
    ff = getattr(args, "ffn_mult", 2.5)
    import math as _math
    if not isinstance(ff, (int, float)) or not _math.isfinite(ff) or ff <= 0:
        raise ValueError(f"--ffn_mult must be positive finite, got {ff!r}")

def _tok(args, out: Path):
    from tokenizer import VERSION as _TOK_VERSION

    tp = Path(args.tokenizer) if args.tokenizer else out / "tokenizer.json"
    if tp.exists():
        try:
            from tokenizer import load as _load
            t = _load(tp)
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"could not load tokenizer {tp}: {exc}") from exc
        if t.get_vocab_size() == args.vocab and t.data.get("version") == _TOK_VERSION:
            return t, tp
        print("[tok] vocab/version mismatch, rebuilding")
    else:
        print(f"[tok] {tp} not found, training new tokenizer")
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise ValueError(f"--data {data_dir} does not exist")
    files = discover_files(data_dir)
    if not files:
        raise RuntimeError(f"no training files found in {data_dir}")
    texts = (t for t, _, _ in iter_texts(files))
    max_records = max(0, int(getattr(args, "tok_records", 0) or 0))
    return ensure_tokenizer(tp, texts, args.vocab, max_records=max_records), tp


def _sha_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_fingerprint(files) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(str(p) for p in files):
        try:
            st = Path(p).stat()
            h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            h.update(p.encode())
    return h.hexdigest()[:16]

def main():
    install_handlers()
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="./datasets")
    a.add_argument("--out", default="./runs/linear")
    a.add_argument("--tokenizer", default=None,
                   help="Tokenizer path (default: <out>/tokenizer.json)")
    a.add_argument("--preset", default=None,
                   help="Named size preset (overrides --vocab/--d/--layers/--heads/--ffn_mult); see --list-presets")
    a.add_argument("--list-presets", action="store_true", help="List size presets with estimated params and exit")
    a.add_argument("--vocab", type=int, default=8000)
    a.add_argument("--d", type=int, default=512)
    a.add_argument("--layers", type=int, default=8)
    a.add_argument("--heads", type=int, default=8)
    a.add_argument("--ffn_mult", type=float, default=2.5)
    a.add_argument("--precision", choices=("fp8", "fp32"), default="fp8")
    a.add_argument("--ctx", type=int, default=256)
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--steps", type=int, default=1000)
    a.add_argument("--lr", type=float, default=2e-4)
    a.add_argument("--wd", type=float, default=0.01)
    a.add_argument("--grad_clip", type=float, default=1.0)
    a.add_argument("--log_every", type=int, default=10)
    a.add_argument("--save_every", type=int, default=200)
    a.add_argument("--tok_records", type=int, default=200000,
                   help="Max records for automatic tokenizer training (0 = unlimited)")
    a.add_argument("--threads", type=int, default=2)
    args = a.parse_args()
    if args.list_presets:
        for _name in sorted(PRESETS):
            _p = PRESETS[_name]
            _est = estimate_params(_p["vocab"], _p["d"], _p["layers"], _p.get("ffn_mult", 2.5))
            print(f"{_name}: vocab={_p['vocab']} d={_p['d']} layers={_p['layers']} "
                  f"heads={_p['heads']} ffn_mult={_p.get('ffn_mult', 2.5)} ~{_est:,} params")
        return
    apply_preset(args)
    _validate_args(args)
    if args.tok_records < 0:
        raise ValueError(f"--tok_records must be non-negative, got {args.tok_records}")
    if args.grad_clip <= 0:
        raise ValueError(f"--grad_clip must be positive, got {args.grad_clip}")
    # Configure threads through the backend (sets OMP/MKL before torch init
    # where possible) instead of duplicating logic here.
    try:
        from compute import get_backend
        get_backend().configure(args.threads)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"invalid --threads {args.threads}: {exc}") from exc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, tok_path = _tok(args, out)
    tok.save(str(tok_path))
    try:
        tok_sha = _sha_file(Path(tok_path))
    except OSError:
        tok_sha = ""
    try:
        ds_fp = _dataset_fingerprint(discover_files(Path(args.data)))
    except (OSError, ValueError, RuntimeError):
        ds_fp = ""
    cfg = LinearConfig(vocab_size=args.vocab, d_model=args.d, n_layer=args.layers, n_heads=args.heads,
                       ffn_mult=getattr(args, "ffn_mult", 2.5),
                       precision=args.precision, tokenizer_sha256=tok_sha, dataset_fingerprint=ds_fp)
    model = SmaulLinear(cfg)
    opt = Lion(list(model.parameters()), lr=args.lr, wd=args.wd, clip=args.grad_clip)
    wrap = load_tokenizer(str(tok_path))
    stream = PretrainStream(Path(args.data), wrap, args.ctx)
    model.train()
    bx, by, step, toks, t0, since = [], [], 0, 0, time.perf_counter(), 0
    bad_steps = 0
    for x, y, _ in stream:
        if STOP or step >= args.steps:
            break
        bx.append(x)
        by.append(y)
        if len(bx) < args.batch:
            continue
        xb, yb = torch.stack(bx), torch.stack(by)
        bx, by = [], []
        opt.zero_grad(model)
        _, loss = model(xb, yb)
        if not torch.isfinite(loss):
            bad_steps += 1
            print(f"[warn] non-finite loss, skip step {step} ({bad_steps} consecutive)")
            opt.zero_grad(model)
            if bad_steps >= 50:
                print("[error] 50 consecutive non-finite losses; stopping to avoid infinite loop")
                break
            continue
        loss.backward()
        norm = opt.step(model)
        if norm == float("inf"):
            bad_steps += 1
            print(f"[warn] non-finite grads, skip step {step} ({bad_steps} consecutive)")
            if bad_steps >= 50:
                print("[error] 50 consecutive non-finite grads; stopping")
                break
            continue
        bad_steps = 0
        step += 1
        toks += xb.numel()
        since += xb.numel()
        if step % args.log_every == 0:
            el = time.perf_counter() - t0
            mem = sum(p.numel() * p.element_size() for p in model.parameters())
            mem += sum(b.numel() * b.element_size() for b in model.buffers())
            # Include Lion momentum: otherwise the metric understates OOM risk.
            mem += sum(v.numel() * v.element_size() for v in opt.m.values())
            print(f"step {step} loss {loss.item():.4f} {since / max(el, 1e-9):.1f} tok/s stored {mem / 1048576:.1f}MiB")
            t0, since = time.perf_counter(), 0
        if step % args.save_every == 0:
            model.save_pretrained(out)
            _save_optimizer(out, opt)
            print(f"[save] {out}")
    if STOP and (bx or by):
        print(f"[stop] dropped {len(bx)} buffered sample(s) from partial batch")
    if step == 0:
        print("[done] no steps completed; checkpoint not overwritten")
        return
    model.save_pretrained(out)
    _save_optimizer(out, opt)
    print(f"[done] steps={step} tokens={toks}")

if __name__ == "__main__":
    main()
