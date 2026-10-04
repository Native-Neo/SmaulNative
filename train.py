#!/usr/bin/env python3
"""SmaulLinear FP8 trainer: pretraining with tiled E4M3 weights, FP32 Lion/SmaulOpt.

Checkpoints are resume-free for Lion (hyperparams only); SmaulOpt
(--optimizer smaul) additionally writes FP32 m/v states for exact resume.
"""
import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import torch

from dataset import PretrainStream, TokenizerWrapper, discover_files, iter_texts
from kernel.fp8_tile import fp8_modules
from model import Lion, PRESETS, SmaulOpt, apply_preset, build_model, estimate_params
from tokenizer import ensure_tokenizer


STOP = False
def _h(sig, fr):
    global STOP
    STOP = True
    print("\n[stop] finishing step then saving")

# Elements per block in the global grad-norm reduction (see _grad_norm).
# 2**19 x 4B = 2 MiB FP32 upcast, which is small enough to stay resident in
# cache and large enough that the Python-level loop is not the bottleneck.

def install_handlers() -> None:
    # Install SIGINT/SIGTERM handlers explicitly from main() only.
    # Importing train (e.g. benchmark.py imports optimizers) must not
    # hijack process signals as a side effect.
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _h)
        except (OSError, ValueError):
            pass



def _save_optimizer(out: Path, opt, model=None) -> None:
    # JSON, not pickle: torch.save would be arbitrary-code-exec on load.
    (out / "optimizer.json").write_text(json.dumps(opt.state_dict(), indent=2), encoding="utf-8")
    if isinstance(opt, SmaulOpt) and model is not None:
        _save_smaul_states(out, opt, model)


def _save_smaul_states(out: Path, opt: "SmaulOpt", model) -> None:
    from safetensors.torch import save_file
    param_names = {id(p): n for n, p in model.named_parameters()}
    mod_names = {id(m): n for n, m in fp8_modules(model)}
    tensors = {}

    def _base(key_obj):
        if id(key_obj) in param_names:
            return f"param.{param_names[id(key_obj)]}"
        if id(key_obj) in mod_names:
            return f"fp8.{mod_names[id(key_obj)]}"
        return None

    for key_obj, m_state in opt.m.items():
        base = _base(key_obj)
        if base is None:
            continue
        # Stored at the checkpoint's own width (bf16/fp16/fp32) so the file size
        # reflects the real state footprint.
        tensors[f"m.{base}"] = m_state.detach().cpu().contiguous()
        st_r = opt.v_row.get(key_obj)
        st_c = opt.v_col.get(key_obj)
        if st_r is not None and st_c is not None:
            # Factored v: the two marginal vectors, never the full [R, C].
            tensors[f"v_row.{base}"] = st_r.detach().cpu().contiguous()
            tensors[f"v_col.{base}"] = st_c.detach().cpu().contiguous()
        else:
            v_state = opt.v.get(key_obj)
            if v_state is not None:
                tensors[f"v.{base}"] = v_state.detach().cpu().contiguous()
    if tensors:
        tmp = out / "optimizer_state.safetensors.tmp"
        save_file(tensors, str(tmp))
        os.replace(tmp, out / "optimizer_state.safetensors")


def _load_optimizer(out: Path, opt, model=None):
    """Load optimizer.json (and SmaulOpt states) into opt.

    Raises ValueError with a clear message when a checkpoint created by
    another optimizer cannot provide the required state.
    """
    out = Path(out)
    data = json.loads((out / "optimizer.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{out / 'optimizer.json'} must contain a JSON object")
    opt.load_state_dict(data)
    if isinstance(opt, SmaulOpt) and model is not None:
        _load_smaul_states(out, opt, model)
    return opt


def _load_smaul_states(out: Path, opt: "SmaulOpt", model) -> None:
    from safetensors.torch import load_file
    out = Path(out)
    sp = out / "optimizer_state.safetensors"
    param_by_name = dict(model.named_parameters())
    mod_by_name = dict(fp8_modules(model))
    if not sp.exists():
        if opt.step_count and (opt.m or opt.v or opt.v_row or opt.v_col):
            raise ValueError(f"SmaulOpt checkpoint {out} is missing optimizer_state.safetensors")
        if int(opt.state_dict().get("step", opt.step_count)) > 0:
            raise ValueError(
                f"SmaulOpt checkpoint {out} records step={opt.step_count} but has no "
                "optimizer_state.safetensors; refusing to pretend state exists")
        return
    try:
        blobs = load_file(str(sp), device="cpu")
    except RuntimeError as exc:
        raise RuntimeError(f"could not load SmaulOpt states {sp}: {exc}") from exc
    new_m: dict = {}
    new_v: dict = {}
    new_vr: dict = {}
    new_vc: dict = {}
    for k, tens in blobs.items():
        if not isinstance(k, str) or "." not in k:
            raise ValueError(f"invalid SmaulOpt state key {k!r}")
        kind, rest = k.split(".", 1)
        if kind not in ("m", "v", "v_row", "v_col"):
            raise ValueError(f"invalid SmaulOpt state key {k!r}")
        if rest.startswith("param."):
            obj = param_by_name.get(rest[len("param."):], None)
            if obj is None:
                raise ValueError(
                    f"SmaulOpt state key {k!r} has no matching parameter in current model")
        elif rest.startswith("fp8."):
            obj = mod_by_name.get(rest[len("fp8."):], None)
            if obj is None:
                raise ValueError(
                    f"SmaulOpt state key {k!r} has no matching FP8 module in current model")
        else:
            raise ValueError(f"invalid SmaulOpt state key {k!r}")
        target = {"m": new_m, "v": new_v, "v_row": new_vr, "v_col": new_vc}[kind]
        target[obj] = tens
    # A model may legitimately hold both forms: 2-D params are factored while
    # 1-D/0-D params keep a full v. What is never valid is one object having
    # both, or a half-written factored pair.
    for obj in new_m:
        has_full = obj in new_v
        r_ok, c_ok = obj in new_vr, obj in new_vc
        if r_ok != c_ok:
            raise ValueError(
                f"SmaulOpt factored state incomplete for one object "
                f"(v_row={r_ok}, v_col={c_ok}); refusing partial load")
        if has_full and r_ok:
            raise ValueError(
                "SmaulOpt state for one object has both full-v and factored-v keys; "
                "refusing to guess")
        if not has_full and not r_ok:
            raise ValueError(f"SmaulOpt state for one object has no v at all; refusing")
    for obj, tens in list(new_vr.items()):
        # Each marginal is 1-D and must line up with its own extent of m:
        # v_row is the mean over dim=1 so it is R long, v_col the mean over
        # dim=0 so it is C long. That is what catches a transposed pair, and
        # for a non-square state it catches it outright.
        #
        # The previous check here compared v_row's shape to v_col's *reversed*
        # shape, which cannot fire: both marginals are 1-D, so `shape[::-1]` is
        # the shape itself, and the two clauses below were mutually exclusive by
        # construction. It was unreachable for every state the optimizer writes,
        # while reading as a guard against exactly this mistake.
        tm = new_m.get(obj)
        if tm is None:
            continue
        if tm.dim() == 2:
            if tens.dim() != 1 or int(tens.shape[0]) != int(tm.shape[0]):
                raise ValueError(
                    f"SmaulOpt v_row has shape {tuple(tens.shape)}, expected ({int(tm.shape[0])},) "
                    f"for an m of {tuple(tm.shape)}; a transposed or mismatched pair")
            tc = new_vc[obj]
            if tc.dim() != 1 or int(tc.shape[0]) != int(tm.shape[1]):
                raise ValueError(
                    f"SmaulOpt v_col has shape {tuple(tc.shape)}, expected ({int(tm.shape[1])},) "
                    f"for an m of {tuple(tm.shape)}; a transposed or mismatched pair")
    # Shape check against live objects (fail clearly on arch change).
    for obj, tm in new_m.items():
        # FP8 modules store [out_f, in_f]; dense params store param shape.
        if hasattr(obj, "_gw"):
            expected = (int(obj.out_f), int(obj.in_f))
        else:
            try:
                expected = tuple(obj.shape)
            except AttributeError:
                expected = None
        if expected is not None and tuple(tm.shape) != expected:
            raise ValueError(
                f"SmaulOpt state shape {tuple(tm.shape)} != expected {expected}; "
                "checkpoint incompatible with current model")
    for obj, tv in new_v.items():
        tm = new_m.get(obj)
        if tm is not None and tuple(tv.shape) != tuple(tm.shape):
            raise ValueError("SmaulOpt m/v shape mismatch; checkpoint corrupt")
    # Cast the restored buffers to the checkpoint's declared storage width so
    # a resumed run keeps the same state footprint and math path.
    want = opt._storage_dtype(signed=True)
    opt.m = {k: v.to(want).contiguous() for k, v in new_m.items()}
    opt.v_row = {k: v.to(want).contiguous() for k, v in new_vr.items()}
    opt.v_col = {k: v.to(want).contiguous() for k, v in new_vc.items()}
    opt.v = {k: v.to(want).contiguous() for k, v in new_v.items()}
    # opt.factor_v keeps whatever the caller configured, so full-vs-factored can
    # be compared without touching source. Two mismatches need a decision:
    if opt.v_row and not opt.factor_v:
        raise ValueError(
            f"SmaulOpt checkpoint {out} holds factored v but factor_v is disabled; "
            "a factored state cannot be expanded into a full v without inventing the "
            "rank term. Resume with factor_v enabled, or start a new run.")
    migrated = 0
    if opt.factor_v and opt.v:
        for key_obj, full in list(opt.v.items()):
            shape = tuple(full.shape)
            if not opt._factor_shape(shape):
                continue
            # Explicit migration, full-v -> factored. The marginals of a stored v
            # are exactly recoverable, so R and C are preserved exactly; only the
            # rank term is dropped, which is what factoring approximates anyway.
            f32 = full.to(want).float()
            opt.v_row[key_obj] = f32.mean(dim=1).to(want).contiguous()
            opt.v_col[key_obj] = f32.mean(dim=0).to(want).contiguous()
            del opt.v[key_obj]
            migrated += 1
        if migrated:
            print(f"[smaul] migrated {migrated} full-v tensor(s) to factored v on load; "
                  "row/col marginals preserved exactly, rank term dropped. "
                  "The next save writes the factored form.")


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
    if getattr(args, "optimizer", "smaul") not in ("lion", "smaul"):
        raise ValueError(f"--optimizer must be lion/smaul, got {getattr(args, 'optimizer')!r}")
    for _n, _flag in (("beta_m", "--beta-m"), ("beta_v", "--beta-v")):
        _v = getattr(args, _n, None)
        if _v is None:
            continue
        if not isinstance(_v, (int, float)) or not math.isfinite(float(_v)):
            raise ValueError(f"{_flag} must be finite, got {_v!r}")
        if not 0.0 <= float(_v) < 1.0:
            raise ValueError(f"{_flag} must be in [0.0, 1.0), got {_v!r}")
    _eps = getattr(args, "epsilon", None)
    if _eps is not None:
        if not isinstance(_eps, (int, float)) or not math.isfinite(float(_eps)) \
                or float(_eps) <= 0:
            raise ValueError(f"--epsilon must be positive finite, got {_eps!r}")
    ff = getattr(args, "ffn_mult", 2.5)
    if not isinstance(ff, (int, float)) or not math.isfinite(ff) or ff <= 0:
        raise ValueError(f"--ffn_mult must be positive finite, got {ff!r}")
    # A negative --rawr-max-docs is truthy, so build_graph's `if max_docs and
    # n_docs >= max_docs` fires on the first document and the graph is built
    # from a single doc. A negative --rawr-max-tokens-per-doc reaches
    # `ids[:max_tokens_per_doc]` and silently drops the last token of every
    # document. Both must fail here, not corrupt the graph quietly.
    for _n in ("rawr_max_docs", "rawr_max_tokens_per_doc"):
        _v = getattr(args, _n, 0)
        if _v is None:
            continue
        if not isinstance(_v, int) or isinstance(_v, bool) or _v < 0:
            raise ValueError(f"--{_n.replace('_', '-')} must be a non-negative int, got {_v!r}")
    # Sparsity and min-degree were validated late (in LinearConfig.__post_init__
    # and build_graph, after the tokenizer was trained and the graph built).
    # Failing here keeps a typo from costing a tokenizer build first.
    _sp = getattr(args, "rawr_sparsity", 0.9)
    if not isinstance(_sp, (int, float)) or isinstance(_sp, bool) \
            or not 0.0 <= float(_sp) < 1.0:
        raise ValueError(f"--rawr-sparsity must be in [0, 1), got {_sp!r}")
    _md = getattr(args, "rawr_min_degree", 4)
    if not isinstance(_md, int) or isinstance(_md, bool) or _md < 1:
        raise ValueError(f"--rawr-min-degree must be a positive int, got {_md!r}")

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
    a.add_argument("--architecture", choices=("rawr", "plain"), default="rawr",
                   help="Model architecture: Rawr sparse (default) or plain dense baseline")
    a.add_argument("--embedding-storage", choices=("ram", "mmap"), default="ram",
                   help="Embedding table storage (default ram)")
    a.add_argument("--rawr-sparsity", type=float, default=0.9,
                   help="Rawr: fraction of hidden/head connections omitted [0, 1). "
                        "This sets K = d_model * (1 - sparsity) columns kept per "
                        "SparseLinear row, so it is really a column-count knob: at the "
                        "old 0.5 default (K = d/2) the [B,T,out_f,K] gather moved 7.7 GiB "
                        "per forward vs 15.3 GiB dense -- half the compute cut while "
                        "still paying every rawr overhead. 0.9 (K ~ d/20) is 5x less "
                        "traffic. Note LinearConfig keeps 0.5 as its code-level default "
                        "so library callers and legacy checkpoints are unaffected.")
    a.add_argument("--rawr-min-degree", type=int, default=4,
                   help="Rawr: fallback connectivity floor per token (>= 1)")
    a.add_argument("--rawr-dict", default=None,
                   help="Rawr: extra dictionary file (one word per line) on top of built-ins")
    a.add_argument("--rawr-graph-out", default=None,
                   help="Rawr: also export the connectivity graph JSON here")
    a.add_argument("--rawr-max-docs", type=int, default=2000,
                   help="Rawr: max corpus docs sampled for graph edges (0 = unlimited)")
    a.add_argument("--rawr-max-tokens-per-doc", type=int, default=1024)
    a.add_argument("--precision", choices=("fp8", "fp32"), default="fp8")
    a.add_argument("--ctx", type=int, default=256)
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--steps", type=int, default=1000)
    a.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (--lr is learning_rate; --wd is weight_decay)")
    a.add_argument("--wd", type=float, default=0.01)
    a.add_argument("--optimizer", choices=("lion", "smaul"), default="smaul",
                   help="Optimizer: smaul (default, SmaulOpt v2.1) or lion "
                        "(resume-free, fixed betas)")
    a.add_argument("--beta-m", dest="beta_m", type=float, default=0.9,
                   help="SmaulOpt beta_m (momentum decay); Lion keeps its built-in betas")
    a.add_argument("--beta-v", dest="beta_v", type=float, default=0.999,
                   help="SmaulOpt beta_v (magnitude-EMA decay)")
    a.add_argument("--epsilon", type=float, default=1e-8,
                   help="SmaulOpt epsilon (positive)")
    a.add_argument("--state-dtype", dest="state_dtype", choices=SmaulOpt._STATE_DTYPES,
                   default="bf16",
                   help="SmaulOpt state storage width; update math is always FP32. "
                        "bf16 (2B, default: ~0.07%% error vs fp32) | fp16 (2B) | "
                        "fp32 (4B, lossless reference)")
    a.add_argument("--grad-dtype", dest="grad_dtype", choices=("bf16", "fp16", "fp32"),
                   default="bf16",
                   help="SmaulOpt: dtype gradients are stored in after backward. "
                        "bf16 (default) halves gradient memory; fp32 keeps them "
                        "wide. Update math is FP32 either way. Lion ignores this.")
    a.add_argument("--factor-v", dest="factor_v", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="SmaulOpt: store v factored (row/col marginals) for 2-D "
                        "parameters instead of full-size. Default on; --no-factor-v "
                        "gives the full-v comparison mode.")
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
            _est = estimate_params(_p["vocab"], _p["d"], _p["layers"], _p["ffn_mult"])
            print(f"{_name}: vocab={_p['vocab']} d={_p['d']} layers={_p['layers']} "
                  f"heads={_p['heads']} ffn_mult={_p['ffn_mult']} ~{_est:,} params")
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
        from kernel.compute import get_backend
        get_backend().configure(args.threads)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"invalid --threads {args.threads}: {exc}") from exc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, tok_path = _tok(args, out)
    # The tokenizer file is written only after the first successful step (see
    # the training loop): writing it here would clobber a good checkpoint's
    # tokenizer on a run that completes zero steps, while the step == 0 path
    # below reports the checkpoint was not overwritten. The sha uses the exact
    # serialization save() writes, so the recorded fingerprint still matches
    # the file once written.
    import hashlib as _hashlib
    tok_sha = _hashlib.sha256(json.dumps(
        tok.data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    try:
        ds_fp = _dataset_fingerprint(discover_files(Path(args.data)))
    except (OSError, ValueError, RuntimeError):
        ds_fp = ""
    opt_name = getattr(args, "optimizer", "smaul") or "smaul"
    model = build_model(args, tok, out, tok_sha, ds_fp)
    if opt_name == "smaul":
        opt = SmaulOpt(list(model.parameters()), lr=args.lr,
                       beta_m=getattr(args, "beta_m", 0.9),
                       beta_v=getattr(args, "beta_v", 0.999),
                       epsilon=getattr(args, "epsilon", 1e-8),
                       weight_decay=args.wd, clip=args.grad_clip,
                       state_dtype=getattr(args, "state_dtype", "bf16"),
                       factor_v=getattr(args, "factor_v", True),
                       grad_dtype=getattr(args, "grad_dtype", "bf16"))
    else:
        opt = Lion(list(model.parameters()), lr=args.lr, wd=args.wd, clip=args.grad_clip)
    wrap = TokenizerWrapper(tok)
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
        # Release the FP32 gradient buffers before the step; the update math is
        # still FP32 (it widens per block). No-op for Lion.
        if hasattr(opt, "narrow_grads_"):
            opt.narrow_grads_(model)
        # Sampled here, not at log time: opt.step() clears every FP8 _gw, so
        # afterwards the live-gradient figure would only see p.grad and
        # understate the peak by the largest single allocation in the model.
        grad_bytes = (sum(p.grad.numel() * p.grad.element_size()
                          for p in model.parameters() if p.grad is not None)
                      + sum(m._gw.numel() * m._gw.element_size()
                            for _, m in fp8_modules(model) if m._gw is not None))
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
        if step == 1:
            tok.save(str(tok_path))
        toks += xb.numel()
        since += xb.numel()
        if step % args.log_every == 0:
            el = time.perf_counter() - t0
            mem = sum(p.numel() * p.element_size() for p in model.parameters())
            mem += sum(b.numel() * b.element_size() for b in model.buffers())
            # Include optimizer state: otherwise the metric understates OOM risk.
            mem += sum(v.numel() * v.element_size() for v in opt.m.values())
            for _store in ("v", "v_row", "v_col"):
                # v is the full-size form; v_row/v_col are the factored
                # marginals, and with --factor-v (the default) v is empty, so
                # counting only v reported almost none of the state.
                _s = getattr(opt, _store, None)
                if _s:
                    mem += sum(v.numel() * v.element_size() for v in _s.values())
            print(f"step {step} loss {loss.item():.4f} {since / max(el, 1e-9):.1f} tok/s "
                  f"stored {mem / 1048576:.1f}MiB (+{grad_bytes / 1048576:.1f}MiB live grads)")
            t0, since = time.perf_counter(), 0
        if step % args.save_every == 0:
            model.save_pretrained(out)
            _save_optimizer(out, opt, model)
            print(f"[save] {out}")
    if STOP and (bx or by):
        print(f"[stop] dropped {len(bx)} buffered sample(s) from partial batch")
    if step == 0:
        print("[done] no steps completed; checkpoint not overwritten")
        return
    model.save_pretrained(out)
    _save_optimizer(out, opt, model)
    print(f"[done] steps={step} tokens={toks}")

if __name__ == "__main__":
    main()
