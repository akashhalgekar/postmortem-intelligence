"""SQLite cache for finished briefs (step B3, the performance layer).

The key covers everything that changes the answer: the normalized question,
retrieval settings, the index fingerprint (so new data invalidates old
answers), the LLM model and the prompt version.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def make_key(**parts) -> str:
    parts["query"] = normalize(parts["query"])
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class BriefCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS briefs (key TEXT PRIMARY KEY, value TEXT, created REAL)")

    def get(self, key: str) -> dict | None:
        with sqlite3.connect(self.path) as c:
            row = c.execute("SELECT value FROM briefs WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO briefs VALUES (?,?,?)", (key, json.dumps(value), time.time()))

    def clear(self) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute("DELETE FROM briefs")

    def __len__(self) -> int:
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT COUNT(*) FROM briefs").fetchone()[0]
