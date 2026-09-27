"""Embedding + FAISS vector store (steps A3 to A5 in the architecture)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from .ingest import doc_text


@lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = _load_model(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes inner product == cosine similarity,
        # which is why the index below is IndexFlatIP.
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")


def fingerprint(records: list[dict], embed_fields: list[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(json.dumps(embed_fields).encode())
    for r in records:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()[:16]


@dataclass
class VectorStore:
    index: faiss.Index
    records: list[dict]
    fingerprint: str
    model_name: str
    embed_fields: list[str]

    @property
    def size(self) -> int:
        return self.index.ntotal

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, records: list[dict], embed_fields: list[str], embedder: Embedder) -> "VectorStore":
        texts = [doc_text(r, embed_fields) for r in records]
        empty = [r["id"] for r, t in zip(records, texts) if not t.strip()]
        if empty:
            raise ValueError(f"Records with no text in embed_fields {embed_fields}: {empty[:10]}")
        vecs = embedder.encode(texts)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index, records, fingerprint(records, embed_fields, embedder.model_name),
                   embedder.model_name, embed_fields)

    # ---------------------------------------------------------------- persist
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        (d / "records.json").write_text(json.dumps(self.records, indent=1))
        (d / "manifest.json").write_text(json.dumps({
            "fingerprint": self.fingerprint, "model_name": self.model_name,
            "embed_fields": self.embed_fields, "size": self.size}, indent=1))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore | None":
        d = Path(directory)
        if not (d / "manifest.json").exists():
            return None
        m = json.loads((d / "manifest.json").read_text())
        index = faiss.read_index(str(d / "index.faiss"))
        records = json.loads((d / "records.json").read_text())
        return cls(index, records, m["fingerprint"], m["model_name"], m["embed_fields"])

    # ---------------------------------------------------------------- search
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = max(1, min(k, self.size))
        scores, idx = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]
