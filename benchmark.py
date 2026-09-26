#!/usr/bin/env python3
"""SmaulNative CPU benchmarks: kernel/training and architecture comparisons.

Use --mode full for FP8/FP32 and end-to-end measurements, or --mode arch for
Rawr/Plain x RAM/mmap comparison.
"""

#!/usr/bin/env python3
"""cpu/benchmark_full.py -- Benchmark the CURRENT SmaulLinear/FP8 pipeline.

Measures separately: FP8Linear forward/backward, Linear Attention, FFN,
RMSNorm, residual add, optimizer/requant, complete training step,
end-to-end tokens/sec, RSS, parameter storage. FP8 vs FP32 side by side.
Do not assume FP8 is faster; this script measures it.

Usage: python cpu/benchmark_full.py [--d 512] [--layers 4] [--ctx 256] [--batch 2] [--iters 10]
"""
import argparse
import resource
import signal
import statistics
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark the CURRENT SmaulLinear/FP8 pipeline")
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    for name in ("d", "layers", "heads", "ctx", "batch", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.d % args.heads != 0:
        raise ValueError(f"--d ({args.d}) must be divisible by --heads ({args.heads})")
    # Guard against accidental OOM: batch*ctx*d floats ~4 bytes each, x3 for grads.
    if args.batch * args.ctx * args.d > 32_000_000:
        raise ValueError("batch*ctx*d too large; reduce --batch/--ctx/--d to avoid OOM")
    return args


def med(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB, macOS returns bytes.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def run_full(argv=None):
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    args = parse_args(argv)

    import torch

    from kernel.compute import get_backend
    from kernel.fp8_tile import FP8Linear, decode_tile, fp8_modules
    from smaul_linear import Block, LinearConfig, SmaulLinear, SwiFFN  # noqa: F401
    # train.py installs a process SIGINT handler at import; save/restore so
    # importing the benchmark as a library has no global side effect.
    _prev_sigint = signal.getsignal(signal.SIGINT)
    try:
        from train import Lion
    finally:
        try:
            signal.signal(signal.SIGINT, _prev_sigint)
        except (OSError, ValueError):
            pass

    be = get_backend()
    be.configure(args.threads)
    torch.manual_seed(42)

    print(f"backend={be.name} native={be.has_native} threads={args.threads} d={args.d} layers={args.layers} ctx={args.ctx}")
    R, D = args.batch * args.ctx, args.d
    out = {}

    torch.manual_seed(0)
    m8 = FP8Linear(D, D)
    m8.train()
    ref = torch.nn.Linear(D, D, bias=False)
    with torch.no_grad():
        nt = (D + 63) // 64
        ref.weight.copy_(torch.cat([decode_tile(m8.w8, m8.sc, 0, D, t) for t in range(nt)], 1))
    x = torch.randn(R, D)
    g = torch.randn(R, D)
    out["fp8_fwd"] = med(lambda: m8(x), args.iters)
    out["fp32_fwd"] = med(lambda: ref(x), args.iters)

    def fp8_bwd():
        xx = x.clone().requires_grad_(True)
        m8(xx).backward(g)

    out["fp8_bwd"] = med(fp8_bwd, args.iters)

    def ref_bwd():
        xx = x.clone().requires_grad_(True)
        ref(xx).backward(g)

    out["fp32_bwd"] = med(ref_bwd, args.iters)

    cfg = LinearConfig(vocab_size=2000, d_model=D, n_layer=1, n_heads=args.heads)
    blk = Block(cfg).eval()
    xb = torch.randn(args.batch, args.ctx, D).to(torch.bfloat16)
    out["attention"] = med(lambda: blk.att(blk.n1(xb)), args.iters)
    out["ffn"] = med(lambda: blk.ffn(xb), args.iters)
    out["rmsnorms"] = med(lambda: (blk.n1(xb), blk.n2(xb.float()), blk.n3(xb), blk.n4(xb.float()), blk.n5(xb)), args.iters)
    a = torch.randn_like(xb)
    out["residual"] = med(lambda: (xb.float() + a.float()).to(xb.dtype), args.iters)

    model = SmaulLinear(LinearConfig(vocab_size=2000, d_model=D, n_layer=args.layers, n_heads=args.heads))
    model.train()
    opt = Lion(list(model.parameters()), lr=2e-4)
    ids = torch.randint(0, 2000, (args.batch, args.ctx))

    def full_step():
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)

    # Time the full step BEFORE the requant micro-bench mutates weights,
    # and use median (not mean) consistently with the other ops.
    def _one_full_step_ms():
        t0 = time.perf_counter()
        full_step()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(3):
        full_step()
    step_ms = statistics.median(_one_full_step_ms() for _ in range(args.iters))
    tps = ids.numel() / (step_ms / 1e3) if step_ms > 0 else 0.0

    # Requant bench on a detached clone so the timed model above is untouched.
    import copy
    m8r = copy.deepcopy(fp8_modules(model)[0][1])
    upd = torch.randn(m8r.out_f, m8r.in_f) * 1e-4
    out["requant"] = med(lambda: m8r.requant(upd, 0.0), args.iters)

    # Stored bytes: FP8 weights (u8) + scales (fp32) + norms/embed/head (fp32/bf16 actual).
    fp8b = sum(m.w8.numel() + m.sc.numel() * 4 for _, m in fp8_modules(model))
    fpb = sum(m.w8.numel() * 4 for _, m in fp8_modules(model))
    other = sum(p.numel() * p.element_size() for p in model.parameters()) - fpb
    # fpb already counts FP8 weights as fp32; other adds the rest.
    print(f"\n{'op':12s} {'FP8 ms':>9s} {'FP32 ms':>9s} {'ratio':>6s}")
    fwd_ratio = out['fp8_fwd'] / out['fp32_fwd'] if out['fp32_fwd'] else float('nan')
    bwd_ratio = out['fp8_bwd'] / out['fp32_bwd'] if out['fp32_bwd'] else float('nan')
    print(f"{'linear_fwd':12s} {out['fp8_fwd']:9.2f} {out['fp32_fwd']:9.2f} {fwd_ratio:6.2f}x")
    print(f"{'linear_bwd':12s} {out['fp8_bwd']:9.2f} {out['fp32_bwd']:9.2f} {bwd_ratio:6.2f}x")
    for k in ("attention", "ffn", "rmsnorms", "residual", "requant"):
        print(f"{k:12s} {out[k]:9.2f}")
    print(f"\nfull step {step_ms:.0f}ms | {tps:.0f} tok/s | RSS {rss_mb():.0f}MB | "
          f"stored FP8 {fp8b/1048576:.1f}MiB vs FP32 {fpb/1048576:.1f}MiB (+other {other/1048576:.1f}MiB)")

#!/usr/bin/env python3
"""Reproducible Rawr-vs-Plain x RAM-vs-mmap comparison (experimental).

Runs all four combos with IDENTICAL dims, tokenizer, dataset, optimizer and
batch config, then reports: param count, stored bytes, graph edges/sparsity,
tokens/sec, peak memory, embedding lookup, startup/load time, train/val loss.

Hypothesis under test (not a claim): Rawr may use the same nominal budget
more effectively because excluded connections consume no storage/compute.

Usage:
    python cpu/benchmark_arch.py [--out ./runs/arch_bench] [--steps 8]
"""
import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

# Fixed tiny experiment (same for all four combos).
DIMS = dict(vocab_size=64, d_model=32, n_layer=2, n_heads=2, ffn_mult=2.0)
CTX, BATCH, STEPS, LR = 32, 2, 8, 1e-4
SEED = 1234

TRAIN_TEXTS = [
    "hello world this is a tiny training sentence for benchmarking",
    "namaste duniya yah ek chhota prashikshan vakya hai",
    "the quick brown fox jumps over 12 lazy dogs and runs code",
    "def train(model): return model.forward(tokens)  # tiny code sample",
    "ek do teen chaar paanch chhah saat aath nau das ginati",
    "language model sparsity test with numbers 123 and punctuation!",
    "hindi bhasha mein yah doosra vakya hai shabd shabd",
    "embedding lookup benchmark row access pattern test test test",
]
VAL_TEXTS = [
    "hello world unseen validation sentence here",
    "namaste validation vakya yahan hai",
]


def build_tokenizer():
    from tokenizer import _build

    texts = list(TRAIN_TEXTS) + list(VAL_TEXTS)
    # NOTE: _build always emits the guaranteed single-char set first, so the
    # realized vocab is larger than small requests; the model dims follow it.
    data = _build(iter(texts), DIMS["vocab_size"], 20000, 0)
    from tokenizer import SmaulTokenizer

    tok = SmaulTokenizer(data)
    DIMS["vocab_size"] = tok.get_vocab_size()
    return tok


def encode_all(tok, texts, ctx):
    out = []
    for t in texts:
        ids = list(tok.encode(t).ids) if hasattr(tok.encode(t), "ids") else list(tok.encode(t))
        ids = ids + [tok.eos_token_id]
        out.extend(ids)
    # Chunk into ctx+1 windows (shared, identical order for every combo).
    chunks = []
    for i in range(0, len(out) - ctx, ctx):
        c = out[i:i + ctx + 1]
        if len(c) == ctx + 1:
            chunks.append(c)
    return chunks


def run_combo(arch, storage, tok, chunks, val_chunks, workdir: Path):
    from smaul_linear import LinearConfig, SmaulLinear
    from train import Lion

    torch.manual_seed(SEED)
    cfg = LinearConfig(precision="fp32", architecture=arch,
                       embedding_storage=storage, rawr_sparsity=0.5,
                       rawr_min_degree=2, **DIMS)
    d = workdir / f"{arch}_{storage}"
    d.mkdir(parents=True, exist_ok=True)
    emb_path = d / "embeddings.dat" if storage == "mmap" else None

    graph = None
    if arch == "rawr":
        from rawr_graph import build_graph

        graph = build_graph(tok, corpus_texts=list(TRAIN_TEXTS),
                            dict_words=None, window=1, min_degree=2)

    t0 = time.perf_counter()
    model = SmaulLinear(cfg, rawr_graph=graph, emb_path=emb_path)
    model.train()
    opt = Lion(list(model.parameters()), lr=LR)
    startup_s = time.perf_counter() - t0

    # Embedding lookup micro-benchmark (row-based random access).
    ids_probe = torch.randint(0, DIMS["vocab_size"], (BATCH, CTX))

    def _lookup():
        with torch.no_grad():
            return model.emb(ids_probe)

    _lookup()
    ts = []
    for _ in range(200):
        a = time.perf_counter()
        _lookup()
        ts.append((time.perf_counter() - a) * 1e6)
    lookup_us = statistics.median(ts)

    train_losses, toks, t_start = [], 0, time.perf_counter()
    step = 0
    bi = 0
    while step < STEPS:
        xb = torch.tensor(chunks[bi % len(chunks)][:-1]).unsqueeze(0).repeat(BATCH, 1)
        yb = torch.tensor(chunks[bi % len(chunks)][1:]).unsqueeze(0).repeat(BATCH, 1)
        bi += 1
        opt.zero_grad(model)
        _, loss = model(xb, yb)
        loss.backward()
        opt.step(model)
        train_losses.append(float(loss.detach()))
        toks += xb.numel()
        step += 1
    train_s = time.perf_counter() - t_start
    tps = toks / max(train_s, 1e-9)

    model.eval()
    with torch.no_grad():
        vl = []
        for c in val_chunks:
            xb = torch.tensor([c[:-1]])
            yb = torch.tensor([c[1:]])
            _, loss = model(xb, yb)
            vl.append(float(loss))
    val_loss = sum(vl) / max(1, len(vl))

    model.save_pretrained(d / "ckpt")
    t1 = time.perf_counter()
    m2 = SmaulLinear.from_pretrained(d / "ckpt")
    load_s = time.perf_counter() - t1

    stored = sum(p.stat().st_size for p in (d / "ckpt").glob("*") if p.is_file())
    n_state = sum(v.numel() for v in model.state_dict().values())
    n_param = sum(p.numel() for p in model.parameters())
    graph_edges = len(graph.edges) if graph is not None else 0
    graph_sp = graph.stats()["sparsity"] if graph is not None else 0.0
    return {
        "arch": arch, "storage": storage,
        "state_values": n_state, "trainable_params": n_param,
        "stored_bytes": stored,
        "graph_edges": graph_edges, "graph_sparsity": graph_sp,
        "tokens_per_sec": tps, "peak_rss_mb": rss_mb(),
        "emb_lookup_us_median": lookup_us,
        "startup_s": startup_s, "load_s": load_s,
        "train_losses": train_losses,
        "final_train_loss": train_losses[-1],
        "val_loss": val_loss,
    }


def run_arch(argv=None):
    a = argparse.ArgumentParser(description="Rawr/Plain x RAM/mmap benchmark")
    a.add_argument("--out", default="./runs/arch_bench")
    a.add_argument("--steps", type=int, default=STEPS)
    args = a.parse_args(argv)
    globals()["STEPS"] = args.steps

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    chunks = encode_all(tok, TRAIN_TEXTS, CTX)
    val_chunks = encode_all(tok, VAL_TEXTS, CTX)
    assert chunks and val_chunks, "no data chunks (tokenizer/dataset issue)"

    results = {}
    for arch, storage in [("rawr", "ram"), ("rawr", "mmap"),
                          ("plain", "ram"), ("plain", "mmap")]:
        print(f"[bench] {arch}+{storage} ...", flush=True)
        results[f"{arch}+{storage}"] = run_combo(arch, storage, tok, chunks,
                                                 val_chunks, out)
    (out / "results.json").write_text(json.dumps(
        {"dims": DIMS, "ctx": CTX, "batch": BATCH, "steps": STEPS,
         "seed": SEED, "results": results}, indent=2))

    hdr = f"{'combo':12s} {'state':>8s} {'stored':>9s} {'edges':>7s} {'spars':>6s} " \
          f"{'tok/s':>8s} {'rssMB':>7s} {'look_us':>8s} {'load_s':>7s} " \
          f"{'train_loss':>10s} {'val_loss':>9s}"
    print("\n" + hdr)
    for k, r in results.items():
        print(f"{k:12s} {r['state_values']:8d} {r['stored_bytes']:9d} "
              f"{r['graph_edges']:7d} {r['graph_sparsity'] * 100:5.1f}% "
              f"{r['tokens_per_sec']:8.0f} {r['peak_rss_mb']:7.0f} "
              f"{r['emb_lookup_us_median']:8.1f} {r['load_s']:7.3f} "
              f"{r['final_train_loss']:10.4f} {r['val_loss']:9.4f}")
    rr, pr = results["rawr+ram"]["val_loss"], results["plain+ram"]["val_loss"]
    print(f"\n[hypothesis check] rawr+ram val {rr:.4f} vs plain+ram val {pr:.4f} "
          f"after {STEPS} identical steps "
          f"({'rawr lower' if rr < pr else 'plain lower or tie'}; "
          f"treat as one tiny data point, not a general claim)")


def main():
    import sys
    mode = "full"
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--mode requires full or arch")
        mode = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    if mode == "full":
        run_full(sys.argv[1:])
    elif mode == "arch":
        run_arch(sys.argv[1:])
    else:
        raise SystemExit(f"unknown --mode {mode!r}; expected full or arch")


if __name__ == "__main__":
    main()
