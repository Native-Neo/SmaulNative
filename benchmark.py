#!/usr/bin/env python3
"""SmaulNative CPU benchmarks: kernel/training and architecture comparisons.

Use --mode full for FP8/FP32 and end-to-end measurements, or --mode arch for
Rawr/Plain x RAM/mmap comparison.
"""

import argparse
import resource
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
    # Default to what train.py actually runs. LinearConfig.architecture
    # defaults to "plain" in code while the CLI defaults to "rawr", so a
    # benchmark that does not pass this measures a configuration nobody
    # trains -- and the rawr sparse path is where the cost is.
    p.add_argument("--arch", choices=("rawr", "plain"), default="rawr",
                   help="architecture to benchmark (default: rawr, the train.py default)")
    p.add_argument("--step-optimizer", dest="step_optimizer",
                   choices=("smaul", "lion"), default="smaul",
                   help="optimizer for the full-step timing in --mode full "
                        "(default: smaul, the train.py default)")
    p.add_argument("--rawr-sparsity", type=float, default=0.9,
                   help="rawr-sparsity used when --arch rawr (default: 0.9, the train.py default)")
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
    """Peak RSS in MiB. ru_maxrss is a high-water mark, not current usage."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB, macOS returns bytes.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def run_full(argv=None):
    _root = str(Path(__file__).resolve().parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    args = parse_args(argv)

    import torch

    from kernel.compute import get_backend
    from kernel.fp8_tile import FP8Linear, decode_tile, fp8_modules
    from model import Block, LinearConfig, Lion, SmaulOpt, SmaulLinear  # noqa: F401

    be = get_backend()
    be.configure(args.threads)
    torch.manual_seed(42)

    print(f"backend={be.name} native={be.has_native} threads={args.threads} d={args.d} "
          f"layers={args.layers} ctx={args.ctx} arch={args.arch} "
          f"step_optimizer={args.step_optimizer}")
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
    # Both forwards under no_grad: ref carries a requires_grad weight and x does
    # not, so without this only the FP32 side builds and saves an AddmmBackward
    # node. That cost lands entirely in fp32_fwd and biases the fwd_ratio this
    # file exists to report. (The FP8 side cannot build one -- FP8Linear's
    # weights are buffers, not parameters.)
    with torch.no_grad():
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

    def mk(n_layer=1, **kw):
        return LinearConfig(vocab_size=2000, d_model=D, n_layer=n_layer,
                            n_heads=args.heads, architecture=args.arch,
                            rawr_sparsity=args.rawr_sparsity, **kw)
    cfg = mk()
    # Block is constructed directly here, so --arch rawr needs the graph that
    # SmaulLinear would otherwise build for itself.
    bench_graph = None
    if args.arch == "rawr":
        from rawr_graph import fallback_graph
        bench_graph = fallback_graph(cfg.vocab_size, cfg.rawr_min_degree)
    blk = Block(cfg, bench_graph).eval()
    xb = torch.randn(args.batch, args.ctx, D).to(torch.bfloat16)
    out["attention"] = med(lambda: blk.att(blk.n1(xb)), args.iters)
    out["ffn"] = med(lambda: blk.ffn(xb), args.iters)
    out["rmsnorms"] = med(lambda: (blk.n1(xb), blk.n2(xb.float()), blk.n3(xb), blk.n4(xb.float()), blk.n5(xb)), args.iters)
    a = torch.randn_like(xb)
    out["residual"] = med(lambda: (xb.float() + a.float()).to(xb.dtype), args.iters)

    model = SmaulLinear(mk(n_layer=args.layers))
    model.train()
    if args.step_optimizer == "smaul":
        opt = SmaulOpt(list(model.parameters()), lr=2e-4)
    else:
        opt = Lion(list(model.parameters()), lr=2e-4)
    ids = torch.randint(0, 2000, (args.batch, args.ctx))

    def full_step():
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        if hasattr(opt, "narrow_grads_"):
            opt.narrow_grads_(model)
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

    # Stored bytes: FP8 weights (u8) + scales (fp32) + everything else at its
    # real width (embedding/head/norms, and SparseLinear.values for rawr).
    # "other" must come from state_dict(), NOT from parameters() minus the FP8
    # weights: FP8Linear stores w8/sc as *buffers*, so the FP8 bytes are not in
    # parameters() at all and that subtraction used to print a negative number.
    fp8b = sum(m.w8.numel() + m.sc.numel() * 4 for _, m in fp8_modules(model))
    fpb = sum(m.w8.numel() * 4 for _, m in fp8_modules(model))
    stored = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    other = stored - fp8b
    print(f"\n{'op':12s} {'FP8 ms':>9s} {'FP32 ms':>9s} {'ratio':>6s}")
    fwd_ratio = out['fp8_fwd'] / out['fp32_fwd'] if out['fp32_fwd'] else float('nan')
    bwd_ratio = out['fp8_bwd'] / out['fp32_bwd'] if out['fp32_bwd'] else float('nan')
    print(f"{'linear_fwd':12s} {out['fp8_fwd']:9.2f} {out['fp32_fwd']:9.2f} {fwd_ratio:6.2f}x")
    print(f"{'linear_bwd':12s} {out['fp8_bwd']:9.2f} {out['fp32_bwd']:9.2f} {bwd_ratio:6.2f}x")
    for k in ("attention", "ffn", "rmsnorms", "residual", "requant"):
        print(f"{k:12s} {out[k]:9.2f}")
    print(f"\nfull step {step_ms:.0f}ms | {tps:.0f} tok/s | RSS {rss_mb():.0f}MB | "
          f"checkpoint {stored/1048576:.1f}MiB = FP8 {fp8b/1048576:.1f}MiB "
          f"(same weights in fp32: {fpb/1048576:.1f}MiB) + {other/1048576:.1f}MiB other")

def run_opt(argv=None):
    """Lion vs SmaulOpt on identical tensors/conditions.

    Update time, persistent optimizer-state bytes, and CPU throughput. Same
    shapes, same gradients, same clipping, same thread count for both; no
    per-optimizer tuning. A measurement, not a claim about convergence.
    """
    _root = str(Path(__file__).resolve().parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    args = parse_args(argv)

    import torch

    from kernel.compute import get_backend
    from model import LinearConfig, Lion, SmaulOpt, SmaulLinear  # noqa: F401

    be = get_backend()
    be.configure(args.threads)
    torch.manual_seed(42)

    print(f"backend={be.name} native={be.has_native} threads={args.threads} d={args.d} "
          f"layers={args.layers} ctx={args.ctx} batch={args.batch} arch={args.arch}")
    cfg = LinearConfig(vocab_size=2000, d_model=args.d, n_layer=args.layers,
                       n_heads=args.heads, architecture=args.arch,
                       rawr_sparsity=args.rawr_sparsity)
    ids = torch.randint(0, 2000, (args.batch, args.ctx))
    # Identical shapes/grads/clip/threads for every entry; the only difference is
    # the optimizer. SmaulOpt is compared across state storage width and across
    # factored vs full v, since that is the memory/latency tradeoff.
    cases = [
        ("lion", None, None),
        ("smaul", "bf16", True), ("smaul", "bf16", False),
        ("smaul", "fp16", True), ("smaul", "fp16", False),
        ("smaul", "fp32", True), ("smaul", "fp32", False),
    ]
    _note = {"bf16": "default", "fp32": "reference"}
    results = {}
    for name, sdt, fv in cases:
        key = name if sdt is None else f"{name}_{sdt}" + ("_factored" if fv else "_fullv")
        torch.manual_seed(1234)
        model = SmaulLinear(cfg)
        model.train()
        if name == "lion":
            opt = Lion(list(model.parameters()), lr=2e-4, clip=1.0)
        else:
            opt = SmaulOpt(list(model.parameters()), lr=2e-4, clip=1.0,
                           state_dtype=sdt, factor_v=bool(fv))

        def one_step():
            opt.zero_grad(model)
            _, loss = model(ids, ids)
            loss.backward()
            return float(opt.step(model))

        for _ in range(3):
            one_step()
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            one_step()
            ts.append((time.perf_counter() - t0) * 1e3)
        step_ms = statistics.median(ts)
        # Persistent optimizer state only (no params, no grads, no transients).
        m_bytes = sum(t.numel() * t.element_size() for t in opt.m.values())
        v_bytes = 0
        for store in ("v", "v_row", "v_col"):
            v_bytes += sum(t.numel() * t.element_size()
                           for t in getattr(opt, store, {}).values())
        state_bytes = m_bytes + v_bytes
        n_state = sum(t.numel() for t in opt.m.values())
        # CPU throughput: identical work per step, so tok/s tracks step cost.
        reps = max(1, args.iters)
        t0 = time.perf_counter()
        for _ in range(reps):
            one_step()
        wall = max(time.perf_counter() - t0, 1e-9)
        tok_s = ids.numel() * reps / wall
        results[key] = {"step_ms": step_ms, "state_bytes": state_bytes,
                        "m_bytes": m_bytes, "v_bytes": v_bytes,
                        "state_values": n_state, "tok_s": tok_s}
        print(f"{key:11s} step {step_ms:8.2f} ms | state {state_bytes / 1048576:7.2f} MiB "
              f"({n_state} values) | {tok_s:8.1f} tok/s")

    fullv = results["smaul_bf16_fullv"]
    # "vs fullv" is relative to bf16 full-v, so every bf16 row reads 1.00x and
    # only the wider-state rows move. It isolates the factored-v saving rather
    # than the fp32 reference.
    print(f"\n{'optimizer':20s} {'step ms':>9s} {'m MiB':>8s} {'v MiB':>8s} "
          f"{'total MiB':>10s} {'vs bf16 fullv':>13s} {'tok/s':>8s}  note")
    for name, v in results.items():
        parts = name.split("_")
        sdt = parts[1] if len(parts) > 1 else ""
        note = _note.get(sdt, "")
        if name.endswith("_fullv"):
            note = (note + " full-v").strip()
        elif name.endswith("_factored"):
            note = (note + " factored-v").strip()
        rel = v["state_bytes"] / max(fullv["state_bytes"], 1)
        print(f"{name:20s} {v['step_ms']:9.2f} {v['m_bytes'] / 1048576:8.2f} "
              f"{v['v_bytes'] / 1048576:8.2f} {v['state_bytes'] / 1048576:10.2f} "
              f"{rel:13.2f}x {v['tok_s']:8.1f}  {note}")
    f_rel = fullv["state_bytes"] / max(results["smaul_bf16_factored"]["state_bytes"], 1)
    print(f"\nbf16 factored-v vs full-v: {f_rel:.2f}x less optimizer state")
    print("(measurement only. Per-step ms is dominated by forward/backward and is "
          "noisy on this machine; the m/v/total MiB columns are exact.)")


# The arch-mode comparison below runs all four Rawr/Plain x RAM/mmap combos
# with IDENTICAL dims, tokenizer, dataset, optimizer and batch config, then
# reports param count, stored bytes, graph edges/sparsity, tokens/sec, peak
# memory, embedding lookup, startup/load time and train/val loss. Hypothesis
# under test, not a claim: Rawr may use the same nominal budget more
# effectively because excluded connections consume no storage/compute.
import json

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
    from model import LinearConfig, Lion, SmaulLinear

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
    # Fail here rather than three lines into run_combo's while loop, where a
    # non-positive value skips the loop entirely and then indexes the empty
    # train_losses list.
    if args.steps <= 0:
        a.error("--steps must be >= 1")
    globals()["STEPS"] = args.steps

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    chunks = encode_all(tok, TRAIN_TEXTS, CTX)
    val_chunks = encode_all(tok, VAL_TEXTS, CTX)
    if not chunks or not val_chunks:
        raise RuntimeError("no data chunks (tokenizer/dataset issue)")

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
    mode = "full"
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--mode requires full, opt, or arch")
        mode = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    if mode == "full":
        run_full(sys.argv[1:])
    elif mode == "opt":
        run_opt(sys.argv[1:])
    elif mode == "arch":
        run_arch(sys.argv[1:])
    else:
        raise SystemExit(f"unknown --mode {mode!r}; expected full, opt, or arch")

if __name__ == "__main__":
    main()
