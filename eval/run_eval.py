"""Retrieval evaluation.

Measures whether the right past incidents come back for a new issue, and
compares the embedding retriever against a plain keyword (TF-IDF) baseline.
Also checks that unrelated questions fall below the similarity floor, which
is what stops the system from producing advice with no real evidence.

Run:  python eval/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postmortem_intel.engine import Engine  # noqa: E402
from postmortem_intel.ingest import doc_text  # noqa: E402

# Questions with nothing to do with the post-mortem library.
OFF_TOPIC = [
    "What is a good recipe for banana bread?",
    "Who won the football match last night?",
    "Suggest names for our new office cat",
    "How do I change my desktop wallpaper?",
    "Write a birthday message for my colleague",
    "What is the capital of Australia?",
]


def metrics(ranked: list[list[str]], relevant: list[list[str]], k: int) -> dict:
    rec, hit1, rr = [], [], []
    for r, rel in zip(ranked, relevant):
        rel = set(rel)
        rec.append(len(set(r[:k]) & rel) / len(rel))
        hit1.append(1.0 if r and r[0] in rel else 0.0)
        first = next((i for i, x in enumerate(r) if x in rel), None)
        rr.append(0.0 if first is None else 1 / (first + 1))
    return {f"recall@{k}": np.mean(rec), "hit@1": np.mean(hit1), "MRR": np.mean(rr)}


def main(k: int = 4) -> dict:
    qs = yaml.safe_load((ROOT / "eval/queries.yaml").read_text())
    eng = Engine(str(ROOT / "config.yaml"))
    store = eng.load_or_build()
    ids = [r["id"] for r in store.records]
    relevant = [q["relevant"] for q in qs]
    unknown = {i for rel in relevant for i in rel} - set(ids)
    assert not unknown, f"eval labels reference unknown IDs: {unknown}"

    # --- embedding retriever (no similarity floor, pure ranking)
    emb_ranked, top1 = [], []
    for q in qs:
        ev = eng.retrieve(q["query"], k=len(ids), min_similarity=-1, relative=False)
        emb_ranked.append([e.id for e in ev])
        top1.append(ev[0].score)

    # --- keyword baseline
    texts = [doc_text(r, eng.cfg["data"]["embed_fields"]) for r in store.records]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts)
    D = vec.transform(texts)
    kw_ranked = []
    for q in qs:
        s = (vec.transform([q["query"]]) @ D.T).toarray()[0]
        kw_ranked.append([ids[i] for i in np.argsort(-s)])

    m_emb = metrics(emb_ranked, relevant, k)
    m_kw = metrics(kw_ranked, relevant, k)

    off = [eng.retrieve(q, k=1, min_similarity=-1, relative=False)[0].score for q in OFF_TOPIC]
    floor = eng.cfg["retrieval"]["min_similarity"]

    print(f"Corpus: {len(ids)} post-mortems | {len(qs)} labelled queries | k={k}")
    print(f"Embedding model: {eng.cfg['retrieval']['embedding_model']}\n")
    print(f"{'metric':<12}{'embeddings':>12}{'TF-IDF':>10}")
    for key in m_emb:
        print(f"{key:<12}{m_emb[key]:>12.3f}{m_kw[key]:>10.3f}")
    print(f"\nTop-1 similarity, on-topic queries : min {min(top1):.3f}  median {np.median(top1):.3f}")
    print(f"Top-1 similarity, off-topic queries: max {max(off):.3f}  median {np.median(off):.3f}")
    print(f"Configured floor (min_similarity)  : {floor:.2f}")
    print(f"  on-topic queries kept   : {sum(s >= floor for s in top1)}/{len(top1)}")
    print(f"  off-topic queries refused: {sum(s < floor for s in off)}/{len(off)}")

    # --- evidence actually passed to the brief (floor + relative margin applied)
    P, Rc, n = [], [], []
    for q in qs:
        kept = [e.id for e in eng.retrieve(q["query"], k=k)]
        rel = set(q["relevant"])
        P.append(len(set(kept) & rel) / len(kept) if kept else 0.0)
        Rc.append(len(set(kept) & rel) / len(rel))
        n.append(len(kept))
    print(f"\nEvidence passed to the brief (floor {floor}, margin {eng.cfg['retrieval'].get('relative_margin')}):")
    print(f"  precision {np.mean(P):.3f}  recall {np.mean(Rc):.3f}  avg incidents {np.mean(n):.2f}")

    print("\nPer-query misses (relevant incidents not in top k):")
    any_miss = False
    for q, r in zip(qs, emb_ranked):
        miss = [x for x in q["relevant"] if x not in r[:k]]
        if miss:
            any_miss = True
            print(f"  - {q['query'][:70]}...\n      missed {miss}, got {r[:k]}")
    if not any_miss:
        print("  none")
    return {"emb": m_emb, "kw": m_kw, "top1": top1, "off": off}


if __name__ == "__main__":
    main()
