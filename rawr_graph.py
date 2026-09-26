#!/usr/bin/env python3
"""Rawr token-connectivity graph.

Builds a deterministic sparse token graph from English/Hindi lexical seed
resources plus the actual SmaulNative training corpus, tokenized with the
actual SmaulNative tokenizer.

Workflow:
    English/Hindi dictionaries (+ built-in seeds)
    -> tokenize words with the SmaulNative tokenizer
    -> valid token-sequence (consecutive-pair) relationships
    -> combine with corpus bigram relationships
    -> sparse connectivity graph (+ configurable fallback)
    -> used by Rawr sparse computation (see smaul_linear.RawrFFN).

The graph is descriptive, not prescriptive: fallback connectivity (self-loop
+ ring neighbours up to ``min_degree``) guarantees every legitimate token
stays reachable even if absent from the dictionary/corpus. Nothing is
permanently pruned from the language; the graph only prioritizes which
hidden connections Rawr materializes.

Determinism: no RNG anywhere. Dictionary words are sorted, corpus texts must
be supplied in a deterministic order (``build`` sorts files via
``dataset.discover_files``), edges are stored sorted, and the digest covers
the canonical edge list plus configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Built-in lexical seeds (small, deterministic; NOT a full dictionary).
# Real deployments should pass --dict-file with larger word lists; the corpus
# always contributes the observed relationships on top.
# ---------------------------------------------------------------------------

EN_SEED_WORDS = sorted(set("""
the be to of and a in that have it for not on with he as you do at this but his
by from they we say her she or an will my one all would there their what so up
out if about who get which go me when make can like time just him know take
person into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even
new want because any these give day most us hello world code data train model
token word text number name function class return import def for while if else
""".split()))

HI_SEED_WORDS = sorted(set("""
hai ki aur ka ke ko ne se mein par vah yah jo to kya nahin hain tha the thi
ho ga ge gi kar sakte samay naam pani ghar kaam bada chhota naya purana din
raat namaste dhanyavad kripya hindi bhasha shabd sankhya ginati ek do teen
chaar paanch chhah saat aath nau das sau hazaar pyaar dost parivaar vidyalaya
""".split()))

# Script-agnostic fallbacks: digits (ASCII + Devanagari), punctuation, and
# code tokens. These guarantee names/numbers/punct/code/multilingual text keep
# representation even with zero corpus overlap.
FALLBACK_TOKENS = sorted(set(
    [str(i) for i in range(10)]
    + [chr(cp) for cp in range(0x0966, 0x0970)]
    + list(".,!?;:()'\"-_/\\|@#$%^&*+=<>{}[]~`")
    + ["def", "class", "return", "import", "for", "while", "if", "else", "0", "1"]
))


def _encode(tokenizer, text: str) -> List[int]:
    out = tokenizer.encode(text)
    return [int(x) for x in list(out)]


def _check_ids(ids: Sequence[int], vocab_size: int, where: str) -> List[int]:
    clean = []
    for i in ids:
        ii = int(i)
        if not 0 <= ii < vocab_size:
            raise ValueError(f"invalid token id {ii} from {where} (vocab {vocab_size})")
        clean.append(ii)
    return clean


@dataclass
class RawrGraph:
    """Deterministic sparse token graph over [0, vocab_size)."""

    vocab_size: int
    # Undirected canonical edges (a <= b), sorted, unique, self-loops allowed.
    edges: List[Tuple[int, int]] = field(default_factory=list)
    window: int = 1
    min_degree: int = 4
    dict_words: int = 0
    corpus_docs: int = 0
    corpus_tokens: int = 0
    corpus_bigrams: int = 0
    dict_coverage: float = 0.0
    corpus_coverage: float = 1.0
    digest: str = ""

    def directed_set(self) -> set:
        s = set()
        for a, b in self.edges:
            s.add((a, b))
            s.add((b, a))
        return s

    def out_degree(self) -> List[int]:
        deg = [0] * self.vocab_size
        for a, b in self.edges:
            if a == b:
                deg[a] += 1
            else:
                deg[a] += 1
                deg[b] += 1
        return deg

    def stats(self) -> Dict:
        v = self.vocab_size
        directed = self.directed_set()
        rawr_count = len(directed)
        dense = v * v
        deg = self.out_degree()
        sp = 1.0 - (rawr_count / dense) if dense else 0.0
        return {
            "vocab_size": v,
            "dense_connection_count": dense,
            "rawr_connection_count": rawr_count,
            "undirected_edge_count": len(self.edges),
            "sparsity": sp,
            "avg_connections_per_token": (sum(deg) / v) if v else 0.0,
            "min_connections": min(deg) if deg else 0,
            "max_connections": max(deg) if deg else 0,
            "dictionary_words": self.dict_words,
            "dictionary_coverage": self.dict_coverage,
            "corpus_docs": self.corpus_docs,
            "corpus_tokens": self.corpus_tokens,
            "corpus_bigram_types": self.corpus_bigrams,
            "corpus_coverage": self.corpus_coverage,
            "estimated_graph_bytes": len(self.edges) * 8,
            "estimated_compute_reduction": sp,
            "digest": self.digest,
            "window": self.window,
            "min_degree": self.min_degree,
        }


def _digest_edges(vocab_size: int, edges: List[Tuple[int, int]],
                  window: int, min_degree: int) -> str:
    h = hashlib.sha256()
    h.update(f"v={vocab_size};w={window};m={min_degree};".encode())
    for a, b in edges:
        h.update(f"{a},{b};".encode())
    return h.hexdigest()[:16]


def build_graph(tokenizer,
                corpus_texts: Optional[Iterable[str]] = None,
                dict_words: Optional[Iterable[str]] = None,
                window: int = 1,
                min_degree: int = 4,
                max_docs: int = 0,
                max_tokens_per_doc: int = 4096) -> RawrGraph:
    """Build the graph. No RNG; fully deterministic for identical inputs."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_degree < 1:
        raise ValueError(f"min_degree must be >= 1, got {min_degree}")
    vocab_size = int(tokenizer.get_vocab_size())
    if vocab_size <= 0:
        raise ValueError(f"bad vocab size {vocab_size}")

    words: List[str] = []
    if dict_words is not None:
        words.extend(w for w in dict_words if isinstance(w, str) and w.strip())
    else:
        words.extend(EN_SEED_WORDS)
        words.extend(HI_SEED_WORDS)
        words.extend(FALLBACK_TOKENS)
    words = sorted(set(w.strip() for w in words if w.strip()))

    edge_set: set = set()
    covered_words = 0
    for w in words:
        try:
            ids = _check_ids(_encode(tokenizer, w), vocab_size, f"dict word {w!r}")
        except (ValueError, RuntimeError):
            continue
        if not ids:
            continue
        ok = True
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            edge_set.add((min(a, b), max(a, b)))
        # Single-token words contribute a self-loop (the token is usable).
        if len(ids) == 1:
            edge_set.add((ids[0], ids[0]))
        # Word is "covered" when every consecutive pair made it into the set.
        if all((min(ids[i], ids[i + 1]), max(ids[i], ids[i + 1])) in edge_set
               for i in range(len(ids) - 1)):
            covered_words += 1

    corpus_bigrams: set = set()
    n_docs = n_toks = 0
    if corpus_texts is not None:
        for text in corpus_texts:
            if max_docs and n_docs >= max_docs:
                break
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                ids = _check_ids(_encode(tokenizer, text), vocab_size, "corpus")
            except (ValueError, RuntimeError):
                continue
            if max_tokens_per_doc and len(ids) > max_tokens_per_doc:
                ids = ids[:max_tokens_per_doc]
            if not ids:
                continue
            n_docs += 1
            n_toks += len(ids)
            for i in range(len(ids) - 1):
                a, b = ids[i], ids[i + 1]
                edge = (min(a, b), max(a, b))
                corpus_bigrams.add(edge)
                edge_set.add(edge)
                if window >= 2 and i + 2 < len(ids):
                    a2, b2 = ids[i], ids[i + 2]
                    edge2 = (min(a2, b2), max(a2, b2))
                    corpus_bigrams.add(edge2)
                    edge_set.add(edge2)

    # Fallback connectivity: self-loop + ring neighbours until min_degree.
    # Guarantees no legitimate token is ever unreachable.
    deg: Dict[int, int] = {}
    for a, b in edge_set:
        if a == b:
            deg[a] = deg.get(a, 0) + 1
        else:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    for v in range(vocab_size):
        d = deg.get(v, 0)
        if d >= min_degree:
            continue
        if (v, v) not in edge_set:
            edge_set.add((v, v))
            d += 1
        k = 1
        while d < min_degree:
            for nb in ((v + k) % vocab_size, (v - k) % vocab_size):
                if d >= min_degree:
                    break
                e = (min(v, nb), max(v, nb))
                if e not in edge_set:
                    edge_set.add(e)
                    d += 1
            k += 1
            if k > vocab_size + 1:  # degenerate tiny vocabs
                break
        deg[v] = d

    edges = sorted(edge_set)
    digest = _digest_edges(vocab_size, edges, window, min_degree)
    corpus_cov = 1.0
    if corpus_bigrams:
        have = sum(1 for e in corpus_bigrams if e in edge_set)
        corpus_cov = have / len(corpus_bigrams)
    return RawrGraph(
        vocab_size=vocab_size,
        edges=[(int(a), int(b)) for a, b in edges],
        window=window,
        min_degree=min_degree,
        dict_words=len(words),
        corpus_docs=n_docs,
        corpus_tokens=n_toks,
        corpus_bigrams=len(corpus_bigrams),
        dict_coverage=(covered_words / len(words)) if words else 0.0,
        corpus_coverage=corpus_cov,
        digest=digest,
    )


def fallback_graph(vocab_size: int, min_degree: int = 4) -> RawrGraph:
    """Ring-fallback-only graph (no dictionary/corpus). Deterministic.

    Used when Rawr needs structure but no corpus is available (unit tests,
    fresh tiny models). Every token gets a self-loop plus ring neighbours up
    to ``min_degree``, so nothing is ever unreachable.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if min_degree < 1:
        raise ValueError(f"min_degree must be >= 1, got {min_degree}")
    edge_set: set = set()
    for v in range(vocab_size):
        edge_set.add((v, v))
        k = 1
        d = 1
        while d < min_degree:
            for nb in ((v + k) % vocab_size, (v - k) % vocab_size):
                if d >= min_degree:
                    break
                e = (min(v, nb), max(v, nb))
                if e not in edge_set:
                    edge_set.add(e)
                    d += 1
            k += 1
            if k > vocab_size + 1:
                break
    edges = sorted(edge_set)
    return RawrGraph(vocab_size=vocab_size, edges=edges, window=0,
                     min_degree=min_degree, digest=_digest_edges(
                         vocab_size, edges, 0, min_degree))


def hidden_cols(out_f: int, in_f: int, graph: RawrGraph,
                sparsity: float, min_per_row: int = 1):
    """Derive deterministic sparse column indices for a hidden linear layer.

    Maps hidden index -> vocab id via modulo, keeps an edge-weighted top-K
    per output row (K from sparsity), distance tiebreak, ring fallback.
    Returns LongTensor (out_f, K) with K >= 1. No RNG.
    """
    import torch

    if out_f <= 0 or in_f <= 0:
        raise ValueError(f"out_f/in_f must be positive, got {out_f}/{in_f}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if min_per_row < 1:
        raise ValueError(f"min_per_row must be >= 1, got {min_per_row}")
    v = graph.vocab_size
    directed = graph.directed_set()
    k = int(round(in_f * (1.0 - sparsity)))
    k = max(min_per_row, min(in_f, k))
    cols = []
    for i in range(out_f):
        vi = i % v
        scored = []
        for j in range(in_f):
            vj = j % v
            s = 1 if (vi, vj) in directed else 0
            scored.append((-s, abs(i - j), j))
        scored.sort()
        cols.append([j for _, _, j in scored[:k]])
    return torch.tensor(cols, dtype=torch.long)


def save_graph(graph: RawrGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"vocab_size": graph.vocab_size, "window": graph.window,
                   "min_degree": graph.min_degree},
        "stats": graph.stats(),
        "edges": [[a, b] for a, b in graph.edges],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def load_graph(path: Path) -> RawrGraph:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"could not load Rawr graph {path}: {exc}") from exc
    try:
        cfg, stats, edges = payload["config"], payload.get("stats", {}), payload["edges"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid Rawr graph file {path}: {exc}") from exc
    graph = RawrGraph(vocab_size=int(cfg["vocab_size"]),
                      edges=[(int(a), int(b)) for a, b in edges],
                      window=int(cfg.get("window", 1)),
                      min_degree=int(cfg.get("min_degree", 4)),
                      dict_words=int(stats.get("dictionary_words", 0)),
                      corpus_docs=int(stats.get("corpus_docs", 0)),
                      corpus_tokens=int(stats.get("corpus_tokens", 0)),
                      corpus_bigrams=int(stats.get("corpus_bigram_types", 0)),
                      dict_coverage=float(stats.get("dictionary_coverage", 0.0)),
                      corpus_coverage=float(stats.get("corpus_coverage", 1.0)),
                      digest=str(stats.get("digest", "")))
    expect = _digest_edges(graph.vocab_size, sorted(graph.edges),
                           graph.window, graph.min_degree)
    if graph.digest and graph.digest != expect:
        raise ValueError(f"Rawr graph {path} digest mismatch (file corrupt?)")
    if not graph.digest:
        graph.digest = expect
    return graph


def print_stats(graph: RawrGraph) -> None:
    s = graph.stats()
    print(f"vocab_size:            {s['vocab_size']:,}")
    print(f"dense_connections:     {s['dense_connection_count']:,}")
    print(f"rawr_connections:      {s['rawr_connection_count']:,}")
    print(f"sparsity:              {s['sparsity'] * 100:.2f}%")
    print(f"avg_conns_per_token:   {s['avg_connections_per_token']:.2f}")
    print(f"min_connections:       {s['min_connections']}")
    print(f"max_connections:       {s['max_connections']}")
    print(f"dictionary_words:      {s['dictionary_words']:,}")
    print(f"dictionary_coverage:   {s['dictionary_coverage'] * 100:.2f}%")
    print(f"corpus_docs:           {s['corpus_docs']:,}")
    print(f"corpus_tokens:         {s['corpus_tokens']:,}")
    print(f"corpus_bigram_types:   {s['corpus_bigram_types']:,}")
    print(f"corpus_coverage:       {s['corpus_coverage'] * 100:.2f}%")
    print(f"estimated_graph_bytes: {s['estimated_graph_bytes']:,}")
    print(f"est_compute_reduction: {s['estimated_compute_reduction'] * 100:.2f}%")
    print(f"digest:                {s['digest']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build/inspect the Rawr token graph")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    p.add_argument("--data", default=None, help="training corpus dir (optional)")
    p.add_argument("--dict-file", default=None, help="extra dictionary (one word per line)")
    p.add_argument("--out", default=None, help="write graph JSON here")
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--min-degree", type=int, default=4)
    p.add_argument("--max-docs", type=int, default=2000)
    p.add_argument("--max-tokens-per-doc", type=int, default=1024)
    args = p.parse_args()

    from tokenizer import SmaulTokenizer
    tok = SmaulTokenizer.from_file(args.tokenizer)

    extra = None
    if args.dict_file:
        extra = [ln.strip() for ln in Path(args.dict_file).read_text(
            encoding="utf-8-sig", errors="replace").splitlines() if ln.strip()]

    texts = None
    if args.data:
        from dataset import discover_files, iter_texts
        files = discover_files(Path(args.data))
        texts = (t for t, _, _ in iter_texts(files))

    graph = build_graph(tok, corpus_texts=texts, dict_words=extra,
                        window=args.window, min_degree=args.min_degree,
                        max_docs=args.max_docs,
                        max_tokens_per_doc=args.max_tokens_per_doc)
    print_stats(graph)
    if args.out:
        save_graph(graph, Path(args.out))
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
