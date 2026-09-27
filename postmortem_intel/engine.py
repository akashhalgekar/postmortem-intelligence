"""Ties the pipeline together: load data -> index -> retrieve -> brief -> cache."""
from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass

from . import brief as B
from .cache import BriefCache, make_key
from .config import load_config, resolve
from .index import Embedder, VectorStore, fingerprint
from .ingest import load_records


@dataclass
class AskResult:
    brief: B.Brief
    cache_hit: bool
    seconds: float
    weak_below: float = 0.40

    @property
    def markdown(self) -> str:
        return B.render_markdown(self.brief, self.weak_below)


class Engine:
    def __init__(self, config_path: str = "config.yaml", llm_client=None, data_path: str | None = None):
        self.cfg = load_config(config_path)
        if data_path:  # override data.path from config for this engine
            self.cfg["data"]["path"] = str(Path(data_path).resolve())
        self.embedder = Embedder(self.cfg["retrieval"]["embedding_model"])
        self.cache = BriefCache(resolve(self.cfg, self.cfg["storage"]["cache_path"]))
        self.llm_client = llm_client  # injectable for tests
        self.store: VectorStore | None = None

    # ------------------------------------------------------------ indexing
    def load_or_build(self, data_path: str | None = None, force: bool = False) -> VectorStore:
        d = self.cfg["data"]
        path = resolve(self.cfg, data_path or d["path"])
        records = load_records(path, d["columns"])
        fp = fingerprint(records, d["embed_fields"], self.embedder.model_name)
        index_dir = resolve(self.cfg, self.cfg["storage"]["index_dir"])
        store = None if force else VectorStore.load(index_dir)
        if store is None or store.fingerprint != fp:
            store = VectorStore.build(records, d["embed_fields"], self.embedder)
            store.save(index_dir)
        self.store = store
        return store

    def build_store(self, records: list[dict]) -> VectorStore:
        """Index records in memory without replacing the default library (used by the UI upload tab)."""
        return VectorStore.build(records, self.cfg["data"]["embed_fields"], self.embedder)

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str, k: int | None = None, min_similarity: float | None = None,
                 relative: bool = True, store: VectorStore | None = None) -> list[B.Evidence]:
        if store is None and self.store is None:
            self.load_or_build()
        store = store or self.store
        r = self.cfg["retrieval"]
        k = k or r["top_k"]
        floor = r["min_similarity"] if min_similarity is None else min_similarity
        qv = self.embedder.encode([query])[0]
        hits = store.search(qv, k)
        hits = [(i, s) for i, s in hits if s >= floor]
        # Drop matches far weaker than the best one: they are usually a different kind of problem.
        margin = r.get("relative_margin")
        if hits and margin is not None and relative:
            best = hits[0][1]
            hits = [(i, s) for i, s in hits if s >= best - margin]
        return [B.Evidence(store.records[i]["id"], s, store.records[i]) for i, s in hits]

    # ------------------------------------------------------------ ask
    def ask(self, query: str, k: int | None = None, min_similarity: float | None = None,
            use_llm: bool = True, use_cache: bool = True, store: VectorStore | None = None) -> AskResult:
        t0 = time.time()
        query = query.strip()
        if not query:
            raise ValueError("Describe the new issue first.")
        if store is None and self.store is None:
            self.load_or_build()
        store = store or self.store
        L = self.cfg["llm"]
        key_present = B.api_key(L["api_key_env"]) is not None or self.llm_client is not None
        mode = "llm" if (use_llm and key_present) else "extractive"
        r = self.cfg["retrieval"]
        ck = make_key(query=query, k=k or r["top_k"],
                      floor=r["min_similarity"] if min_similarity is None else min_similarity,
                      index=store.fingerprint, mode=mode, model=L["model"] if mode == "llm" else None,
                      prompt=B.PROMPT_VERSION)
        if use_cache:
            hit = self.cache.get(ck)
            if hit:
                return AskResult(B.Brief.from_dict(hit), True, time.time() - t0, r["weak_match_below"])

        evidence = self.retrieve(query, k, min_similarity, store=store)
        llm_error = None
        if not evidence:
            b = B.Brief(query, "no_evidence", None, {}, [], [])
        else:
            allowed = {e.id for e in evidence}
            all_ids = {rec["id"] for rec in store.records}
            llm_error = None
            if mode == "llm":
                try:
                    raw = B.call_groq(B.build_messages(query, evidence), L["model"], L["temperature"],
                                      B.api_key(L["api_key_env"]), client=self.llm_client)
                except Exception as exc:  # auth, rate limit, network, malformed output
                    llm_error = f"{type(exc).__name__}: {exc}"
                    mode = "extractive"
            if mode == "extractive":
                raw = B.extractive(evidence)
                if llm_error:
                    raw["evidence_gaps"].insert(0, f"LLM call failed, showing extractive brief instead ({llm_error[:200]}).")
            summary, sections, gaps, report = B.validate(raw, allowed, all_ids)
            b = B.Brief(query, mode, summary, sections, gaps, evidence, report,
                        L["model"] if mode == "llm" else None)
        if use_cache and not (evidence and llm_error):
            self.cache.put(ck, b.to_dict())
        return AskResult(b, False, time.time() - t0, r["weak_match_below"])
