#!/usr/bin/env python
"""No-LLM lexical baseline -- the control the paper must beat (or explain).

Resolves each query with pure lexical grounding + structural relations, no
language model at all:

  direct / attribute / ungroundable
      score every target-eligible node against the query's terms with
      verify.grounded(); answer the highest-scoring node that clears the
      threshold; abstain if none does (the discriminator gate makes the
      near-miss stratum abstain, absent categories the far stratum).
  relational
      resolve the anchor lexically (best grounding for the query terms),
      then apply verify.relation_holds() with the query's structured
      relation; answer the unique node that satisfies it, else abstain.

If this scores near the LLM+verify pipeline, the templated benchmark is
lexically trivial and the LLM's value must be shown on harder, paraphrased
queries -- a question to settle before the full grid, not in review.

Writes rows in the same schema as run_ablation.py (model="lexical") so
analyse.py can table it beside the LLM runs, and prints a per-stratum table.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyse import _METRIC_HEADERS, _metric_row, md_table, metrics  # noqa: E402
from experiments.run_ablation import (  # noqa: E402
    COLUMNS, PROMPT_VERSION, THRESHOLD, git_commit, is_correct, query_id,
)
from src.benchmark import _target_rooms, generate, index_by_plan  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402
from src.verify import grounded, relation_holds  # noqa: E402

CONFIG_ID = "lexical|main|n0|v1|elexical|kNA"


def _best_grounding(m, phrase, threshold):
    """Highest-scoring node that grounds ``phrase``, or None."""
    best, best_score = None, -1.0
    for node in sorted(_target_rooms(m), key=lambda n: n.id):
        g = grounded(m, node.id, phrase, threshold)
        if g.ok and g.details["score"] > best_score:
            best, best_score = node.id, g.details["score"]
    return best


def resolve_lexical(query, m, threshold=THRESHOLD):
    """Return (node_id or None). No LLM."""
    phrase = " ".join(query.query_terms) if query.query_terms else query.text
    if query.stratum == "relational":
        anchor = _best_grounding(m, phrase, threshold)
        if anchor is None:
            return None
        relation = getattr(query, "relation", None)
        sat = [n.id for n in _target_rooms(m)
               if n.id != anchor
               and relation_holds(m, n.id, anchor, relation).ok]
        return sat[0] if len(sat) == 1 else None
    # direct / attribute / ungroundable
    return _best_grounding(m, phrase, threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--n-per-stratum", type=int, default=200)
    ap.add_argument("--split", default="tuning")
    ap.add_argument("--strata", default=None,
                    help="comma-separated strata to keep (e.g. 'functional')")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        ROOT / "experiments" / "results" / f"baseline_lexical_seed{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    maps = load_dataset(args.sample, args.seed)
    idx = index_by_plan(maps)
    keep = set(args.strata.split(",")) if args.strata else None
    queries = [q for q in generate(maps, args.n_per_stratum, args.seed,
                                   report=False)
               if q.split == args.split
               and (keep is None or q.stratum in keep)]
    queries.sort(key=query_id)

    meta = {
        "git_commit": git_commit(), "seed": args.seed, "prompt_version": "none",
        "split": args.split, "dataset_sample": args.sample,
        "n_per_stratum": args.n_per_stratum, "n_queries": len(queries),
    }
    rows = []
    with out.open("w", newline="") as f:
        for key, val in meta.items():
            f.write(f"# {key}: {val}\n")
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for q in queries:
            t = time.monotonic()
            node = resolve_lexical(q, idx[q.plan_id])
            dt = time.monotonic() - t
            abstained = node is None
            correct = is_correct(q.stratum, q.expected, node, abstained)
            row = [CONFIG_ID, "main", "lexical", "none", 0, True, "lexical",
                   "NA", 0, THRESHOLD, "none", query_id(q), q.plan_id,
                   q.stratum, q.expected or "", node or "", abstained, correct,
                   1, "" if not abstained else "no_lexical_match", 0, 0, 0,
                   f"{dt:.4f}", False, "",  # latency, widened, verify_predicates
                   "ok",                    # status (lexical never fails infra)
                   ""]                      # current_node
            w.writerow(row)
            rows.append(dict(zip(COLUMNS, [str(x) for x in row])))

    print(f"[lexical baseline] {len(rows)} queries, split={args.split} "
          f"-> {out}\n")
    print(md_table(_METRIC_HEADERS, [_metric_row("lexical (no LLM)",
                                                 metrics(rows))]))


if __name__ == "__main__":
    main()
