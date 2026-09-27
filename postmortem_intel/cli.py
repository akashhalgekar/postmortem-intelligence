"""Command line interface.

  python -m postmortem_intel.cli index                 # build or refresh the FAISS index
  python -m postmortem_intel.cli ask "supplier fire, one week of stock left"
  python -m postmortem_intel.cli ask "..." --json      # machine readable output
  python -m postmortem_intel.cli --data my_incidents.csv ask "..."   # use another file
  python -m postmortem_intel.cli clear-cache
"""
from __future__ import annotations

import argparse
import json
import sys

from .engine import Engine


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pmi", description="Post-Mortem Intelligence")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", help="use this data file or folder instead of data.path in config")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index", help="build the vector index from the data file")
    i.add_argument("--force", action="store_true")

    a = sub.add_parser("ask", help="get a brief for a new issue")
    a.add_argument("query")
    a.add_argument("-k", type=int)
    a.add_argument("--min-similarity", type=float)
    a.add_argument("--no-llm", action="store_true", help="extractive brief only")
    a.add_argument("--no-cache", action="store_true")
    a.add_argument("--json", action="store_true")

    sub.add_parser("clear-cache")
    args = p.parse_args(argv)

    eng = Engine(args.config, data_path=args.data)
    if args.cmd == "index":
        s = eng.load_or_build(force=args.force)
        print(f"Indexed {s.size} post-mortems with {s.model_name} (fingerprint {s.fingerprint})")
    elif args.cmd == "ask":
        r = eng.ask(args.query, k=args.k, min_similarity=args.min_similarity,
                    use_llm=not args.no_llm, use_cache=not args.no_cache)
        if args.json:
            print(json.dumps(r.brief.to_dict() | {"cache_hit": r.cache_hit}, indent=2))
        else:
            print(r.markdown)
            print("\nEvidence:")
            for e in r.brief.evidence:
                print(f"  {e.id}  similarity {e.score:.2f}  {e.record.get('title', '')}")
            print(f"\n({'cache hit' if r.cache_hit else 'computed'} in {r.seconds:.2f}s)")
    elif args.cmd == "clear-cache":
        eng.cache.clear()
        print("Cache cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
