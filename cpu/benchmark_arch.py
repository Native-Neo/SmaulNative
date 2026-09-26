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

_ROOT = str(Path(__file__).resolve().parent.parent)
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


def rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


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


def main():
    a = argparse.ArgumentParser(description="Rawr/Plain x RAM/mmap benchmark")
    a.add_argument("--out", default="./runs/arch_bench")
    a.add_argument("--steps", type=int, default=STEPS)
    args = a.parse_args()
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


if __name__ == "__main__":
    main()
