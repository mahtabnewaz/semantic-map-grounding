#!/usr/bin/env python
"""Compare methods on the functional stratum (tuning split).

Reports, beyond overall accuracy:
  - per-category accuracy for each method;
  - per-category accuracy vs per-category mean lexical grounding score
    (does model advantage track LOW groundability?), with a correlation;
  - abstention counts per method and pairwise overlap;
  - for the lexical baseline: abstain vs answered-wrongly.

Usage: functional_compare.py CSV [CSV ...]   (mix of method result CSVs)
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.functional_queries import FUNCTIONAL_QUERIES  # noqa: E402
from experiments.analyse import read_csv  # noqa: E402
from experiments.run_ablation import query_id  # noqa: E402
from src.benchmark import _target_rooms, generate, index_by_plan  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402
from src.verify import _grounding_score  # noqa: E402


def _pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: functional_compare.py CSV [CSV ...]")
    meta0, _ = read_csv(Path(paths[0]))
    seed = int(meta0["seed"])
    sample = int(meta0["dataset_sample"])
    n_per = int(meta0["n_per_stratum"])
    split = meta0.get("split", "tuning")

    # regenerate the functional queries for THIS split -> id maps
    maps = load_dataset(sample, seed)
    fqs = [q for q in generate(maps, n_per, seed, report=False)
           if q.stratum == "functional" and q.split == split]
    qid_cat = {query_id(q): q.template_id.split(".")[1].split("#")[0]
               for q in fqs}

    # per-category mean groundability (over all authored phrasings)
    rep = {}
    for m in maps:
        for n in _target_rooms(m):
            rep.setdefault(n.label, (m, n.id))
    cat_ground = {}
    for cat, phr in FUNCTIONAL_QUERIES.items():
        sc = [_grounding_score(*rep[cat], p) for p in phr] if cat in rep else []
        cat_ground[cat] = statistics.mean(sc) if sc else float("nan")

    # load methods; exclude invalid rows (infra failures) from scoring
    methods, invalid_ct = {}, {}
    for p in paths:
        _, rows = read_csv(Path(p))
        fr = [r for r in rows if r["stratum"] == "functional"]
        if not fr:
            continue
        name = fr[0]["model"]
        invalid_ct[name] = sum(r.get("status") == "invalid" for r in fr)
        methods[name] = [r for r in fr if r.get("status") != "invalid"]

    cats = sorted({c for c in qid_cat.values()})
    print(f"\n# Functional stratum comparison (split={split}, "
          f"{len(fqs)} queries)\n")

    # -- overall + per-category accuracy (invalid rows excluded) --
    print("## Overall accuracy  (invalid = excluded infra failures)")
    for name, fr in methods.items():
        acc = sum(r["correct"] == "True" for r in fr) / len(fr) if fr else 0
        inv = invalid_ct.get(name, 0)
        tag = f"  [EXCLUDED {inv} invalid]" if inv else ""
        print(f"  {name:26s}: {acc:.3f}  (n={len(fr)} scored){tag}")

    print("\n## Per-category accuracy  (groundability = mean lexical score)")
    header = f"{'category':13s} {'ground':>7s} " + " ".join(
        f"{m.split('/')[-1][:10]:>10s}" for m in methods)
    print(header)
    n_by_cat = Counter(qid_cat.values())
    for cat in cats:
        line = f"{cat:13s} {cat_ground.get(cat, float('nan')):7.3f} "
        for name, fr in methods.items():
            sub = [r for r in fr if qid_cat.get(r["query_id"]) == cat]
            acc = (sum(r["correct"] == "True" for r in sub) / len(sub)
                   if sub else float("nan"))
            line += f"{acc:10.2f}" if acc == acc else f"{'-':>10s}"
        print(line + f"   (N={n_by_cat[cat]})")

    # -- correlation: per-category accuracy vs groundability --
    print("\n## Correlation(per-category accuracy, groundability)")
    print("   (negative => method helps most where lexical route is weakest)")
    for name, fr in methods.items():
        xs, ys = [], []
        for cat in cats:
            sub = [r for r in fr if qid_cat.get(r["query_id"]) == cat]
            if sub and cat_ground.get(cat) == cat_ground.get(cat):
                ys.append(sum(r["correct"] == "True" for r in sub) / len(sub))
                xs.append(cat_ground[cat])
        print(f"  {name:26s}: r = {_pearson(xs, ys):+.3f}")

    # -- abstention counts + overlap --
    print("\n## Abstention counts")
    abst = {}
    for name, fr in methods.items():
        s = {r["query_id"] for r in fr if r["abstained"] == "True"}
        abst[name] = s
        print(f"  {name:26s}: abstains {len(s)}/{len(fr)}")
    names = list(abst)
    if len(names) >= 2:
        print("  pairwise abstention overlap (Jaccard):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = abst[names[i]], abst[names[j]]
                u = len(a | b)
                print(f"    {names[i].split('/')[-1][:12]:12s} & "
                      f"{names[j].split('/')[-1][:12]:12s}: "
                      f"{len(a & b)}/{u} = {len(a & b) / u:.2f}" if u else "0")

    # -- lexical: abstain vs answered-wrongly --
    for name, fr in methods.items():
        if "lexical" in name:
            ab = sum(r["abstained"] == "True" for r in fr)
            wrong = sum(r["abstained"] == "False" and r["correct"] == "False"
                        for r in fr)
            right = sum(r["correct"] == "True" for r in fr)
            print(f"\n## Lexical baseline behaviour on functional (n={len(fr)})")
            print(f"  abstained: {ab}   answered wrongly: {wrong}   "
                  f"answered correctly: {right}")


if __name__ == "__main__":
    main()
