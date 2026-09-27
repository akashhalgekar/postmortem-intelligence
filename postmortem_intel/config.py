"""Loads config.yaml and resolves paths relative to the config file."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

STANDARD_FIELDS = [
    "id", "title", "date", "domain", "category", "severity", "description",
    "root_cause", "contributing_factors", "actions_taken", "outcome",
    "lessons_learned", "kpis_impacted",
]

DEFAULTS = {
    "data": {
        "path": "data/sample_postmortems.csv",
        "columns": {f: f for f in STANDARD_FIELDS} | {"id": "incident_id"},
        "embed_fields": ["title", "category", "description", "root_cause",
                         "contributing_factors", "lessons_learned"],
    },
    "retrieval": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 4,
        "min_similarity": 0.30,
        "weak_match_below": 0.40,
        "relative_margin": 0.15,
    },
    "llm": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "temperature": 0.1,
        "api_key_env": "GROQ_API_KEY",
    },
    "storage": {"index_dir": ".pmi/index", "cache_path": ".pmi/cache.sqlite"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "columns":
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = _merge(DEFAULTS, raw or {})
    # Optional override, e.g. a local folder holding the model for offline use.
    if os.environ.get("PMI_EMBEDDING_MODEL"):
        cfg["retrieval"]["embedding_model"] = os.environ["PMI_EMBEDDING_MODEL"]
    root = path.resolve().parent
    cfg["_root"] = str(root)

    unknown = set(cfg["data"]["columns"]) - set(STANDARD_FIELDS)
    if unknown:
        raise ValueError(f"Unknown standard field(s) in data.columns: {sorted(unknown)}. "
                         f"Allowed: {STANDARD_FIELDS}")
    if not cfg["data"]["columns"].get("id"):
        raise ValueError("data.columns.id is required (the column holding a unique incident ID).")
    bad = [f for f in cfg["data"]["embed_fields"] if f not in STANDARD_FIELDS or f == "id"]
    if bad:
        raise ValueError(f"embed_fields contains invalid field(s): {bad}")
    return cfg


def resolve(cfg: dict, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
