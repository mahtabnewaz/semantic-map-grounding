#!/usr/bin/env python
"""Characterise the frozen functional stratum BEFORE running any model.

Reports, for functional_queries.VERSION:
  A. lexical grounding score of each phrasing vs its OWN target category
     (histogram, mean, median, per-category mean, count scoring 0.0);
  B. generation survivors and ambiguity rejections (by cause, and by
     target>competitor pair).

Nothing here is tuned to a number; it only measures the frozen file.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.functional_queries import FUNCTIONAL_QUERIES, VERSION  # noqa: E402
from src.benchmark import _functional_ops, _target_rooms  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402
from src.verify import _grounding_score  # noqa: E402

SEED, SAMPLE = 0, 400


def main():
    maps = load_dataset(SAMPLE, SEED)
    n_phrasings = sum(len(v) for v in FUNCTIONAL_QUERIES.values())
    print(f"\nfunctional stratum VERSION = {VERSION}")
    print(f"{n_phrasings} phrasings across {len(FUNCTIONAL_QUERIES)} "
          f"categories; {len(maps)} plans (sample={SAMPLE}, seed={SEED})")

    # representative node per category (attributes are category-constant)
    rep = {}
    for m in maps:
        for n in _target_rooms(m):
            rep.setdefault(n.label, (m, n.id))

    # -- A. per-phrasing self-category grounding score --------------------
    print("\n== A. lexical grounding score of each phrasing vs its OWN "
          "category ==")
    all_scores, per_cat = [], {}
    for cat, phrasings in FUNCTIONAL_QUERIES.items():
        scores = ([_grounding_score(*rep[cat], p) for p in phrasings]
                  if cat in rep else [])
        per_cat[cat] = scores
        all_scores += scores

    buckets = {"0.00": 0, "(0.0,0.2]": 0, "(0.2,0.4]": 0,
               "(0.4,0.6]": 0, "(0.6,0.8]": 0, "(0.8,1.0]": 0}
    for s in all_scores:
        if s == 0.0:
            buckets["0.00"] += 1
        elif s <= 0.2:
            buckets["(0.0,0.2]"] += 1
        elif s <= 0.4:
            buckets["(0.2,0.4]"] += 1
        elif s <= 0.6:
            buckets["(0.4,0.6]"] += 1
        elif s <= 0.8:
            buckets["(0.6,0.8]"] += 1
        else:
            buckets["(0.8,1.0]"] += 1
    n = len(all_scores)
    print(f"n scored = {n}   mean = {statistics.mean(all_scores):.3f}   "
          f"median = {statistics.median(all_scores):.3f}")
    print("histogram:")
    for b, c in buckets.items():
        bar = "#" * round(40 * c / n)
        print(f"  {b:10s} {c:3d} ({100 * c / n:4.0f}%) {bar}")
    print(f"phrasings scoring exactly 0.0 (no lexical route): "
          f"{buckets['0.00']} of {n} ({100 * buckets['0.00'] / n:.0f}%)")
    print("per-category mean grounding score (higher = more lexical route):")
    for cat in FUNCTIONAL_QUERIES:
        sc = per_cat[cat]
        m_ = statistics.mean(sc) if sc else float("nan")
        z = sum(1 for x in sc if x == 0.0)
        print(f"  {cat:14s} n={len(sc):2d}  mean={m_:.3f}  zeros={z}")

    # -- B. generation survivors + ambiguity rejections -------------------
    print("\n== B. generation: survivors and ambiguity rejections "
          "(pre-sampling pool over all plans) ==")
    rej = Counter()
    survivors = []
    for m in sorted(maps, key=lambda x: x.plan_id):
        survivors += _functional_ops(m, rej)
    surv_by_cat = Counter(op[2] for op in survivors)
    print(f"surviving functional query instances (plan x phrasing): "
          f"{len(survivors)}")
    print("surviving instances per category:")
    for cat in FUNCTIONAL_QUERIES:
        print(f"  {cat:14s}: {surv_by_cat.get(cat, 0)}")
    print("\nrejections by cause:")
    print(f"  duplicate_category (target not unique in plan): "
          f"{rej['functional_duplicate_category']}")
    print(f"  cross-category (a competitor present grounds it): "
          f"{rej['functional_crosscat']}")
    print("\ncross-category rejections by (target > competitor):")
    pairs = {k: v for k, v in rej.items()
             if isinstance(k, tuple) and k[0] == "functional_crosscat_pair"}
    for (_, t, c), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {t:14s} > {c:14s}: {v}")


if __name__ == "__main__":
    main()
