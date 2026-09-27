"""Gradio front end (step B1).

Run:  python app.py        then open http://127.0.0.1:7860
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from postmortem_intel.config import STANDARD_FIELDS
from postmortem_intel.engine import Engine
from postmortem_intel.ingest import DataError, load_records

ROOT = Path(__file__).resolve().parent
ENGINE = Engine(str(ROOT / "config.yaml"))
ENGINE.load_or_build()

EXAMPLES = [
    "Our only supplier of a critical sensor had a fire and we have one week of stock left.",
    "Leadership wants the new warehouse management system live two weeks before Black Friday.",
    "The customer sent a drawing change three weeks before our PPAP submission.",
    "Our 3PL was hit by ransomware and we cannot see shipment status.",
    "Sales is running a big retailer promotion next month and planning has not seen the volumes.",
]

NONE = "(not in my file)"


def _status(store=None) -> str:
    s = store or ENGINE.store
    src = "your uploaded file" if store else "sample library"
    llm = "on" if os.environ.get(ENGINE.cfg["llm"]["api_key_env"]) else "off (extractive mode, set GROQ_API_KEY)"
    return (f"**Library ({src}):** {s.size} post-mortems indexed with `{Path(s.model_name).name}` · "
            f"**LLM:** {llm} · **Cached briefs:** {len(ENGINE.cache)}")


def ask(query, k, floor, use_llm, store):
    if not query or not query.strip():
        return "Describe the new issue first.", pd.DataFrame(), _status(store)
    try:
        r = ENGINE.ask(query, k=int(k), min_similarity=float(floor), use_llm=use_llm, store=store)
    except Exception as e:  # show the reason instead of a stack trace
        return f"**Error:** {e}", pd.DataFrame(), _status(store)
    rows = [{"ID": e.id, "Similarity": round(e.score, 3), "Title": e.record.get("title", ""),
             "Category": e.record.get("category", ""), "Date": e.record.get("date", "")}
            for e in r.brief.evidence]
    note = f"\n\n---\n*{'Served from cache' if r.cache_hit else 'Computed'} in {r.seconds:.2f}s.*"
    return r.markdown + note, pd.DataFrame(rows), _status(store)


# ---------------------------------------------------------------- own data

_SYNONYMS = {
    "id": ["id", "incident_id", "incident", "ticket", "ticket_id", "case", "case_id", "key", "number"],
    "title": ["title", "headline", "name", "subject", "short_description"],
    "description": ["description", "summary", "what_happened", "details", "incident_description"],
    "root_cause": ["root_cause", "cause", "root_causes", "rca"],
    "contributing_factors": ["contributing_factors", "contributing_factor", "factors"],
    "actions_taken": ["actions_taken", "actions", "fix", "resolution", "corrective_action", "remediation"],
    "outcome": ["outcome", "impact", "result", "consequence"],
    "lessons_learned": ["lessons_learned", "lesson", "lessons", "learnings", "takeaways"],
    "kpis_impacted": ["kpis_impacted", "kpis", "kpi", "metrics", "measures"],
}


def _guess(cols: list[str], field: str) -> str:
    norm = {c: "_".join(c.lower().replace("-", " ").split()) for c in cols}
    for cand in _SYNONYMS.get(field, [field]):
        for c, n in norm.items():
            if n == cand:
                return c
    return NONE


def inspect_upload(file):
    if file is None:
        return [gr.update(choices=[NONE], value=NONE)] * len(STANDARD_FIELDS) + ["Upload a file first."]
    path = Path(file)
    try:
        df = pd.read_csv(path, dtype=str, nrows=5) if path.suffix.lower() == ".csv" else pd.read_excel(path, dtype=str, nrows=5)
    except Exception as e:
        return [gr.update()] * len(STANDARD_FIELDS) + [f"Could not read file: {e}"]
    cols = list(df.columns)
    ups = [gr.update(choices=[NONE] + cols, value=_guess(cols, f)) for f in STANDARD_FIELDS]
    return ups + [f"Found columns: {', '.join(cols)}. Check the mapping below, then build."]


def build_from_upload(file, store, *mapping):
    if file is None:
        return "Upload a file first.", _status(store), store
    cols = {f: ("" if m in (None, NONE) else m) for f, m in zip(STANDARD_FIELDS, mapping)}
    if not cols["id"]:
        return "Map the ID column. Every post-mortem needs a unique ID.", _status(store), store
    try:
        records = load_records(file, cols)
        new_store = ENGINE.build_store(records)
    except (DataError, ValueError) as e:
        return f"**Could not build index:** {e}", _status(store), store
    return (f"Indexed {len(records)} records from {Path(file).name}. Go to the Ask tab.",
            _status(new_store), new_store)


def reset_sample():
    return "Back to the sample library.", _status(None), None


with gr.Blocks(title="Post-Mortem Intelligence") as demo:
    gr.Markdown("# Post-Mortem Intelligence\n"
                "Describe a new operational issue. The system finds the most similar past post-mortems "
                "and writes a brief (risks, root causes, actions, KPIs) that uses **only** that evidence "
                "and cites it.")
    status = gr.Markdown(_status())
    session_store = gr.State(None)  # per-browser-session uploaded library; None = sample library
    with gr.Tab("Ask"):
        q = gr.Textbox(label="New issue", lines=3,
                       placeholder="e.g. Our sole supplier of a sealed connector just had a fire...")
        with gr.Row():
            k = gr.Slider(1, 8, value=ENGINE.cfg["retrieval"]["top_k"], step=1, label="Past incidents to retrieve")
            floor = gr.Slider(0.0, 0.8, value=ENGINE.cfg["retrieval"]["min_similarity"], step=0.05,
                              label="Minimum similarity")
            use_llm = gr.Checkbox(value=True, label="Use LLM (needs GROQ_API_KEY)")
        btn = gr.Button("Generate brief", variant="primary")
        gr.Examples(EXAMPLES, inputs=q)
        out = gr.Markdown()
        ev = gr.Dataframe(label="Retrieved evidence", interactive=False)
        btn.click(ask, [q, k, floor, use_llm, session_store], [out, ev, status])
        q.submit(ask, [q, k, floor, use_llm, session_store], [out, ev, status])

    with gr.Tab("Use your own data"):
        gr.Markdown("Upload a CSV or Excel file of your own post-mortems, incident reports or lessons learned. "
                    "Map your columns to the standard fields. Only **id** and at least one text field are needed. "
                    "The index lives in memory for your browser session only; for a permanent setup edit `config.yaml`.")
        f = gr.File(label="Post-mortem file", file_types=[".csv", ".xlsx"], type="filepath")
        msg = gr.Markdown()
        with gr.Accordion("Column mapping", open=True):
            dds = []
            for start in range(0, len(STANDARD_FIELDS), 4):
                with gr.Row():
                    dds += [gr.Dropdown([NONE], value=NONE, label=fld, allow_custom_value=False)
                            for fld in STANDARD_FIELDS[start:start + 4]]
        with gr.Row():
            b2 = gr.Button("Build index from my file", variant="primary")
            b3 = gr.Button("Back to sample data")
        f.change(inspect_upload, f, dds + [msg])
        b2.click(build_from_upload, [f, session_store] + dds, [msg, status, session_store])
        b3.click(reset_sample, None, [msg, status, session_store])

    gr.Markdown("<small>Sample data is fictional and for demonstration only.</small>")

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
