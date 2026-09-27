"""Loads post-mortem records from CSV / XLSX / JSON / JSONL or a folder of
Markdown/text files, and maps them onto the standard field names.

Every record comes out as a plain dict with all STANDARD_FIELDS present
(empty string when missing) so the rest of the pipeline never has to guess.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import STANDARD_FIELDS


class DataError(ValueError):
    pass


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str).fillna("")
    if suf == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False).astype(str)
    if suf == ".json":
        return pd.read_json(path, dtype=False).astype(str)
    raise DataError(f"Unsupported file type: {suf}. Use CSV, XLSX, JSON, JSONL or a folder of .md/.txt files.")


# Heading words -> standard field, for Markdown post-mortems.
_HEADING_MAP = {
    "summary": "description", "description": "description", "what happened": "description",
    "incident": "description", "impact": "outcome", "outcome": "outcome", "result": "outcome",
    "root cause": "root_cause", "root causes": "root_cause", "cause": "root_cause",
    "contributing factors": "contributing_factors", "contributing factor": "contributing_factors",
    "actions": "actions_taken", "actions taken": "actions_taken", "corrective actions": "actions_taken",
    "resolution": "actions_taken", "remediation": "actions_taken",
    "lessons": "lessons_learned", "lessons learned": "lessons_learned", "learnings": "lessons_learned",
    "kpis": "kpis_impacted", "kpis impacted": "kpis_impacted", "metrics": "kpis_impacted",
    "category": "category", "domain": "domain", "severity": "severity", "date": "date",
}


def parse_markdown(text: str, record_id: str) -> dict:
    rec = {f: "" for f in STANDARD_FIELDS}
    rec["id"] = record_id
    current = "description"
    buf: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(#{1,6})\s+(.*?)\s*$", line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not rec["title"]:
                rec["title"] = heading
                continue
            key = re.sub(r"[^a-z ]", "", heading.lower()).strip()
            current = _HEADING_MAP.get(key, "description")
            continue
        buf.setdefault(current, []).append(line)
    for field, lines in buf.items():
        txt = "\n".join(lines).strip()
        rec[field] = (rec[field] + "\n" + txt).strip() if rec[field] else txt
    if not rec["title"]:
        rec["title"] = record_id
    return rec


def load_records(path: str | Path, columns: dict) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Data path not found: {path}")

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".md", ".txt"))
        if not files:
            raise DataError(f"No .md or .txt files in folder {path}")
        records = [parse_markdown(p.read_text(encoding="utf-8"), p.stem) for p in files]
    else:
        df = _read_table(path)
        df.columns = [str(c).strip() for c in df.columns]
        missing = [src for std, src in columns.items() if src and src not in df.columns]
        if missing:
            raise DataError(
                f"Column(s) {missing} named in config data.columns are not in {path.name}. "
                f"Columns found: {list(df.columns)}")
        records = []
        for _, row in df.iterrows():
            rec = {f: "" for f in STANDARD_FIELDS}
            for std, src in columns.items():
                if src:
                    val = row[src]
                    rec[std] = "" if val is None or str(val).lower() == "nan" else str(val).strip()
            records.append(rec)

    # Validation
    records = [r for r in records if any(r[f] for f in STANDARD_FIELDS if f != "id")]
    if not records:
        raise DataError("No non-empty records found.")
    ids = [r["id"] for r in records]
    if any(not i for i in ids):
        raise DataError("Some records have an empty ID. Every post-mortem needs a unique ID.")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DataError(f"Duplicate IDs found: {dupes[:10]}")
    return records


def doc_text(rec: dict, embed_fields: list[str]) -> str:
    """Combine the chosen fields into one labelled string for embedding."""
    parts = []
    for f in embed_fields:
        if rec.get(f):
            parts.append(f"{f.replace('_', ' ').title()}: {rec[f]}")
    return "\n".join(parts)
