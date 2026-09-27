"""End to end and unit tests. The LLM is replaced by a fake client so tests
run offline and deterministically; the embedding model is real."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from postmortem_intel import brief as B
from postmortem_intel.cli import main as cli_main
from postmortem_intel.engine import Engine
from postmortem_intel.ingest import DataError, load_records, parse_markdown

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_postmortems.csv"


class FakeGroq:
    """Mimics groq.Groq().chat.completions.create and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.responses.pop(0) if self.responses else "{}"
        if callable(content):
            content = content(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_config(tmp_path: Path, data: Path = SAMPLE, **overrides) -> Path:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    shutil.copy(data, tmp_path / data.name)
    cfg["data"]["path"] = data.name
    cfg["storage"] = {"index_dir": "idx", "cache_path": "cache.sqlite"}
    for k, v in overrides.items():
        cfg[k].update(v)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    e = Engine(str(make_config(tmp_path)))
    e.load_or_build()
    return e


# ----------------------------------------------------------------- ingest

def test_sample_data_loads_with_all_fields():
    from postmortem_intel.config import load_config
    cfg = load_config(ROOT / "config.yaml")
    recs = load_records(SAMPLE, cfg["data"]["columns"])
    assert len(recs) == 36
    assert all(r["id"].startswith("PM-") and r["root_cause"] and r["lessons_learned"] for r in recs)


def test_missing_column_gives_clear_error(tmp_path):
    with pytest.raises(DataError, match="not in"):
        load_records(SAMPLE, {"id": "incident_id", "description": "does_not_exist"})


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("id,text\nA,one\nA,two\n")
    with pytest.raises(DataError, match="Duplicate"):
        load_records(p, {"id": "id", "description": "text"})


def test_markdown_folder_ingest(tmp_path):
    d = tmp_path / "pms"
    d.mkdir()
    (d / "INC-7.md").write_text("# Router outage\n\n## Summary\nCore router failed.\n\n"
                                "## Root Cause\nFirmware bug.\n\n## Lessons Learned\nStage firmware rollouts.\n")
    (d / "INC-8.txt").write_text("Printer jammed during label run.")
    recs = load_records(d, {})
    r7 = next(r for r in recs if r["id"] == "INC-7")
    assert r7["title"] == "Router outage"
    assert r7["root_cause"] == "Firmware bug."
    assert r7["lessons_learned"] == "Stage firmware rollouts."
    assert next(r for r in recs if r["id"] == "INC-8")["description"].startswith("Printer")


def test_parse_markdown_unknown_heading_goes_to_description():
    r = parse_markdown("# T\n## Timeline\n09:00 alarm\n", "X")
    assert "09:00 alarm" in r["description"]


# ----------------------------------------------------------------- index

def test_index_persists_and_rebuilds_on_data_change(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = make_config(tmp_path)
    e1 = Engine(str(cfg))
    s1 = e1.load_or_build()
    assert (tmp_path / "idx" / "index.faiss").exists() and s1.size == 36
    e2 = Engine(str(cfg))
    assert e2.load_or_build().fingerprint == s1.fingerprint      # loaded, same data
    # change one record -> fingerprint changes -> index rebuilt
    csv = tmp_path / SAMPLE.name
    csv.write_text(csv.read_text().replace("A fire at the only approved", "A flood at the only approved"))
    assert Engine(str(cfg)).load_or_build().fingerprint != s1.fingerprint


def test_cosine_scores_are_bounded(engine):
    ev = engine.retrieve("supplier fire stopped our line", k=36, min_similarity=-1, relative=False)
    assert len(ev) == 36
    assert all(-1.0001 <= e.score <= 1.0001 for e in ev)
    assert [e.score for e in ev] == sorted((e.score for e in ev), reverse=True)


# ----------------------------------------------------------------- retrieval quality guards

@pytest.mark.parametrize("query,expected", [
    ("Customs is holding our import because the tariff code on the invoice looks wrong", "PM-006"),
    ("Our 3PL got hacked with ransomware and we cannot see any shipment status", "PM-032"),
    ("System says we have stock but the bin is empty", "PM-034"),
])
def test_obvious_matches_rank_first(engine, query, expected):
    assert engine.retrieve(query, k=1)[0].id == expected


def test_off_topic_question_refuses_without_calling_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    fake = FakeGroq([])
    e = Engine(str(make_config(tmp_path)), llm_client=fake)
    r = e.ask("What is a good recipe for banana bread?")
    assert r.brief.mode == "no_evidence"
    assert fake.calls == []
    assert "No sufficiently similar" in r.markdown


# ----------------------------------------------------------------- LLM path and grounding

def _good_llm_reply(kw):
    ids = sorted(set(__import__("re").findall(r'<incident id="([^"]+)"', kw["messages"][1]["content"])))
    first = ids[0]
    return json.dumps({
        "summary": {"text": "Looks like a single source disruption.", "citations": [first]},
        "risks": [{"text": "Line stop within days.", "citations": [first]},
                  {"text": "Made up risk with a fake source.", "citations": ["PM-999"]},
                  {"text": "Uncited opinion.", "citations": []}],
        "root_causes": [{"text": "No qualified alternate supplier.", "citations": [first, "PM-999"]}],
        "actions": [{"text": "Qualify a second source, as in PM-003 which was not retrieved.",
                     "citations": [first]}],
        "kpis": [{"text": "Line uptime", "citations": [first]}],
        "evidence_gaps": ["No data on the sensor supplier's other plants."],
    })


def test_llm_brief_is_grounded_and_invalid_citations_are_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    fake = FakeGroq([_good_llm_reply])
    e = Engine(str(make_config(tmp_path)), llm_client=fake)
    r = e.ask("A fire at our only connector supplier stopped their molding and we will run out", k=2)
    b = r.brief
    retrieved = {x.id for x in b.evidence}
    assert b.mode == "llm" and len(fake.calls) == 1
    # prompt only contains retrieved incidents
    prompt = fake.calls[0]["messages"][1]["content"]
    assert set(__import__("re").findall(r'<incident id="([^"]+)"', prompt)) == retrieved
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    # every surviving item cites retrieved evidence only
    for items in b.sections.values():
        for it in items:
            assert it["citations"] and set(it["citations"]) <= retrieved
    texts = [it["text"] for it in b.sections["risks"]]
    assert "Made up risk with a fake source." not in texts
    assert "Uncited opinion." not in texts
    assert b.sections["root_causes"][0]["citations"] == [next(iter(sorted(retrieved)))]
    assert b.validation["items_dropped_no_valid_citation"] == 2
    assert "PM-999" in b.validation["invalid_citations_removed"]
    if "PM-003" not in retrieved:
        assert "PM-003" in b.validation["uncited_id_mentions"]
    assert "Grounding check" in r.markdown


def test_malformed_json_retries_then_falls_back_without_caching(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    fake = FakeGroq(["not json", "still not json"])
    e = Engine(str(make_config(tmp_path)), llm_client=fake)
    r = e.ask("supplier fire stopped the line")
    assert len(fake.calls) == 2
    assert r.brief.mode == "extractive"
    assert "LLM call failed" in r.brief.evidence_gaps[0]
    assert len(e.cache) == 0          # a fallback must not be cached as if it were the LLM answer


def test_llm_api_error_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "bad")

    def boom(kw):
        raise RuntimeError("401 invalid api key")
    e = Engine(str(make_config(tmp_path)), llm_client=FakeGroq([boom]))
    r = e.ask("supplier fire stopped the line")
    assert r.brief.mode == "extractive" and "401" in r.brief.evidence_gaps[0]


def test_malformed_json_then_valid_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    fake = FakeGroq(["oops", _good_llm_reply])
    e = Engine(str(make_config(tmp_path)), llm_client=fake)
    assert e.ask("supplier fire stopped the line").brief.mode == "llm"


# ----------------------------------------------------------------- extractive mode

def test_extractive_mode_only_quotes_records(engine):
    r = engine.ask("Customs is holding our import because the tariff code looks wrong")
    b = r.brief
    assert b.mode == "extractive"
    by_id = {e.id: e.record for e in b.evidence}
    for key, items in b.sections.items():
        for it in items:
            for c in it["citations"]:
                assert c in by_id
            text = it["text"].removeprefix("Past impact: ")
            assert any(text in v for c in it["citations"] for v in by_id[c].values()), (key, text)


# ----------------------------------------------------------------- cache

def test_cache_hit_and_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    fake = FakeGroq([_good_llm_reply, _good_llm_reply])
    cfg = make_config(tmp_path)
    e = Engine(str(cfg), llm_client=fake)
    q = "Supplier fire stopped the line"
    r1 = e.ask(q)
    r2 = e.ask("  supplier FIRE stopped   the line ")   # same question, different spacing/case
    assert not r1.cache_hit and r2.cache_hit and len(fake.calls) == 1
    assert r2.brief.to_dict() == r1.brief.to_dict()
    e.ask(q, k=2)                                        # different settings -> recompute
    assert len(fake.calls) == 2


# ----------------------------------------------------------------- CLI

def test_cli_ask_json(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = make_config(tmp_path)
    assert cli_main(["--config", str(cfg), "ask", "Our 3PL was hit by ransomware", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "extractive" and out["evidence"][0]["id"] == "PM-032"


# ----------------------------------------------------------------- other industries / own data

def test_other_industry_file_with_different_column_names(engine):
    cols = {"id": "Ticket", "title": "Headline", "description": "What happened", "root_cause": "Cause",
            "actions_taken": "Fix", "lessons_learned": "Lesson", "kpis_impacted": "Metrics"}
    recs = load_records(ROOT / "data/examples/it_incidents_example.csv", cols)
    store = engine.build_store(recs)
    r = engine.ask("Our SSL certificate expires soon and renewal is manual", store=store)
    assert r.brief.evidence[0].id == "INC-102"
    # default library untouched by the uploaded one
    assert engine.store.size == 36 and store.size == 6


def test_cli_data_override_applies_to_ask(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = make_config(tmp_path)
    md = ROOT / "data/examples/markdown"
    assert cli_main(["--config", str(cfg), "--data", str(md), "ask", "dock label printer broke", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [e["id"] for e in out["evidence"]] == ["INC-2024-017"]
