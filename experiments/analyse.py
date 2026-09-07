#!/usr/bin/env python
"""Analyse ablation result CSVs into the paper's markdown tables.

Reads one or more result CSVs (see run_ablation.py) and emits:
  - the main ablation table (per model x (n_max, verify_enabled)),
  - encoding and k_hop sweep tables when those grids are present,
  - a model-comparison table at each model's best main config,
  - a grounding-threshold sweep computed from the TUNING split only.

The threshold sweep is oracle-based and model-independent: for the tuning
split it scores each query's expected node (groundable) or best-matching
room (ungroundable) with verify.grounded() across a range of thresholds, to
show how the grounding threshold trades answer rate against abstention. It
is computed on tuning so the chosen threshold never sees the test split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import _target_rooms, generate, index_by_plan  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402
from src.verify import grounded  # noqa: E402

UNGROUNDABLE = {"ungroundable_far", "ungroundable_near"}
STRATA = ["direct", "attribute", "relational",
          "ungroundable_far", "ungroundable_near", "functional"]


# --------------------------------------------------------------------------
# CSV reading
# --------------------------------------------------------------------------
def read_csv(path: Path):
    meta, rows, header = {}, [], None
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("#"):
                line = row[0][1:].strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    meta[k] = v
                continue
            if header is None:
                header = row
                continue
            rows.append(dict(zip(header, row)))
    return meta, rows


def load_all(paths):
    metas, rows = [], []
    for p in paths:
        m, r = read_csv(Path(p))
        metas.append(m)
        rows.extend(r)
    return metas, rows


# --------------------------------------------------------------------------
# metrics + formatting
# --------------------------------------------------------------------------
def _acc(rows):
    return (sum(r["correct"] == "True" for r in rows) / len(rows)
            if rows else float("nan"))


def metrics(all_rows: list[dict]) -> dict:
    # Exclude invalid rows (transport error / empty reply) from accuracy --
    # they are infrastructure failures, not model answers. Old CSVs without a
    # status column are treated as all-ok.
    invalid = [r for r in all_rows if r.get("status") == "invalid"]
    rows = [r for r in all_rows if r.get("status") != "invalid"]
    m = {"n": len(rows), "invalid": len(invalid)}
    for s in STRATA:
        m[f"acc_{s}"] = _acc([r for r in rows if r["stratum"] == s])
    m["overall"] = _acc(rows)
    abst = [r for r in rows if r["abstained"] == "True"]
    true_abst = [r for r in abst if r["stratum"] in UNGROUNDABLE]
    m["abstain_precision"] = (len(true_abst) / len(abst)
                              if abst else float("nan"))
    for s in ("ungroundable_far", "ungroundable_near"):
        sub = [r for r in rows if r["stratum"] == s]
        m[f"recall_{s}"] = (sum(r["abstained"] == "True" for r in sub)
                            / len(sub) if sub else float("nan"))
    att = [int(r["attempts"]) for r in rows if r["attempts"].isdigit()]
    m["mean_attempts"] = sum(att) / len(att) if att else float("nan")
    lat = [float(r["latency_s"]) for r in rows if r["latency_s"]]
    m["mean_latency"] = sum(lat) / len(lat) if lat else float("nan")
    stok = [int(r["subgraph_tokens"]) for r in rows
            if r["subgraph_tokens"].isdigit()]
    m["subgraph_tokens"] = sum(stok) / len(stok) if stok else float("nan")
    ptok = [int(r["prompt_tokens"]) for r in rows
            if r["prompt_tokens"].isdigit()]
    m["prompt_tokens"] = sum(ptok) / len(ptok) if ptok else float("nan")
    return m


def _f(x, p=3):
    return "-" if x != x else f"{x:.{p}f}"     # x!=x catches NaN


def md_table(headers, rows) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    out = [line(headers), line(["---"] * len(headers))]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _metric_row(label, m):
    return [label, m["n"], m.get("invalid", 0), _f(m["overall"]),
            _f(m["acc_direct"]),
            _f(m["acc_attribute"]), _f(m["acc_relational"]),
            _f(m["acc_ungroundable_far"]), _f(m["acc_ungroundable_near"]),
            _f(m["acc_functional"]),
            _f(m["abstain_precision"]), _f(m["recall_ungroundable_far"]),
            _f(m["recall_ungroundable_near"]), _f(m["mean_attempts"], 2),
            _f(m["mean_latency"], 2), _f(m["subgraph_tokens"], 0),
            _f(m["prompt_tokens"], 0)]


_METRIC_HEADERS = ["config", "n", "inv", "overall", "direct", "attr", "rel",
                   "un_far", "un_near", "func", "abs_prec", "rec_far",
                   "rec_near", "attempts", "latency_s", "sub_tok", "prompt_tok"]


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def best_main(rows, model):
    main = [r for r in rows if r["grid"] == "main" and r["model"] == model]
    groups = defaultdict(list)
    for r in main:
        groups[(int(r["n_max"]), r["verify_enabled"] == "True")].append(r)
    if not groups:
        return None
    return max(groups, key=lambda k: (_acc(groups[k]), -k[0]))


def table_main(rows) -> str:
    main = [r for r in rows if r["grid"] == "main"]
    models = sorted({r["model"] for r in main})
    out = []
    for model in models:
        groups = defaultdict(list)
        for r in main:
            if r["model"] == model:
                groups[(int(r["n_max"]), r["verify_enabled"] == "True")]\
                    .append(r)
        for (n_max, verify) in sorted(groups):
            label = f"{model} n={n_max} v={int(verify)}"
            out.append(_metric_row(label, metrics(groups[(n_max, verify)])))
    return md_table(_METRIC_HEADERS, out)


def table_sweep(rows, grid, dim_col, model) -> str:
    """Encoding or k_hop sweep at the model's best main config, including the
    main-grid baseline (structured / k=2)."""
    best = best_main(rows, model)
    if best is None:
        return ""
    n_max, verify = best
    baseline = [r for r in rows if r["grid"] == "main" and r["model"] == model
                and int(r["n_max"]) == n_max
                and (r["verify_enabled"] == "True") == verify]
    sweep = [r for r in rows if r["grid"] == grid and r["model"] == model
             and int(r["n_max"]) == n_max
             and (r["verify_enabled"] == "True") == verify]
    if not sweep:
        return ""
    groups = defaultdict(list)
    for r in baseline:
        groups[r[dim_col]].append(r)
    for r in sweep:
        groups[r[dim_col]].append(r)
    out = [_metric_row(f"{model} {dim_col}={val}", metrics(groups[val]))
           for val in sorted(groups)]
    return md_table(_METRIC_HEADERS, out)


def table_models(rows) -> str:
    out = []
    for model in sorted({r["model"] for r in rows if r["grid"] == "main"}):
        best = best_main(rows, model)
        if best is None:
            continue
        n_max, verify = best
        sub = [r for r in rows if r["grid"] == "main" and r["model"] == model
               and int(r["n_max"]) == n_max
               and (r["verify_enabled"] == "True") == verify]
        out.append(_metric_row(f"{model} (best n={n_max} v={int(verify)})",
                               metrics(sub)))
    return md_table(_METRIC_HEADERS, out)


# --------------------------------------------------------------------------
# verification coverage + functional stratum
# --------------------------------------------------------------------------
def coverage_section(rows) -> str:
    """Which predicates were applicable per stratum -- so accuracy is never
    read as if every stratum had the same verification available."""
    cov = defaultdict(set)
    for r in rows:
        vp = r.get("verify_predicates", "")
        if vp and vp != "disabled":
            cov[r["stratum"]].add(vp)
    if not cov:
        return ""
    lines = ["## Verification coverage (applicable predicates per stratum)\n"]
    for s in STRATA:
        if cov.get(s):
            lines.append(f"- **{s}**: {' | '.join(sorted(cov[s]))}")
    lines.append("\n_functional has no semantic predicate (grounding has no "
                 "lexical route; a category check would be an oracle), so its "
                 "verification coverage is reduced by design._")
    return "\n".join(lines) + "\n"


def _load_functional(meta):
    """(idx, functional queries) for the CSV's split, regenerated from meta."""
    seed, sample = int(meta["seed"]), int(meta["dataset_sample"])
    n_per, split = int(meta["n_per_stratum"]), meta.get("split", "tuning")
    maps = load_dataset(sample, seed)
    qs = [q for q in generate(maps, n_per, seed, report=False)
          if q.split == split and q.stratum == "functional"]
    return index_by_plan(maps), qs


def functional_report(rows, meta) -> str:
    from experiments.run_ablation import query_id  # same id hash as the runner

    fr = [r for r in rows if r["stratum"] == "functional"]
    if not fr:
        return "_no functional queries present (phrasings not yet authored)._"

    idx, fqs = _load_functional(meta)
    qid_cat = {query_id(q): q.template_id.split(".")[1].split("#")[0]
               for q in fqs}

    lines = []
    # (a) per-method accuracy + abstention at each method's best main config
    method_sub = {}
    tbl = []
    for model in sorted({r["model"] for r in fr}):
        best = best_main(rows, model)
        sub = [r for r in fr if r["model"] == model]
        if best:
            n_max, ver = best
            sub = [r for r in sub if int(r["n_max"]) == n_max
                   and (r["verify_enabled"] == "True") == ver]
        method_sub[model] = sub
        n = len(sub)
        ab = sum(r["abstained"] == "True" for r in sub) / n if n else float("nan")
        tbl.append([model, n, _f(_acc(sub)), _f(ab)])
    lines.append("### accuracy / abstention (best main config per method)\n")
    lines.append(md_table(["method", "n", "accuracy", "abstain_rate"], tbl))

    # (b) per-category accuracy per method
    cats = sorted(set(qid_cat.values()))
    if cats:
        hdr = ["category"] + list(method_sub)
        body = []
        for c in cats:
            row = [c]
            for model, sub in method_sub.items():
                cs = [r for r in sub if qid_cat.get(r["query_id"]) == c]
                row.append(_f(_acc(cs)) if cs else "-")
            body.append(row)
        lines.append("\n### per-category accuracy\n")
        lines.append(md_table(hdr, body))

    # (c) lexical grounding-score distribution across the frozen stratum
    scores = []
    for q in fqs:
        m = idx[q.plan_id]
        best = max((grounded(m, n.id, q.query_terms[0], 0.0).details["score"]
                    for n in _target_rooms(m)), default=0.0)
        scores.append(best)
    if scores:
        scores.sort()
        import statistics
        pct = lambda p: scores[min(len(scores) - 1, int(p * len(scores)))]
        frac = sum(s >= 0.60 for s in scores) / len(scores)
        lines.append("\n### lexical grounding score across the stratum "
                     "(measured, not a filter)\n")
        lines.append(f"- n={len(scores)}  min={scores[0]:.2f}  "
                     f"p25={pct(.25):.2f}  median={statistics.median(scores):.2f}"
                     f"  p75={pct(.75):.2f}  max={scores[-1]:.2f}")
        lines.append(f"- fraction grounding at T=0.60: {frac:.2f} "
                     f"(these are phrasings a lexical matcher CAN reach)")

    # (d) abstention overlap between methods
    if len(method_sub) >= 2:
        ab_sets = {model: {r["query_id"] for r in sub
                           if r["abstained"] == "True"}
                   for model, sub in method_sub.items()}
        models = list(ab_sets)
        lines.append("\n### abstention overlap\n")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                inter = len(ab_sets[a] & ab_sets[b])
                lines.append(f"- {a} abstains {len(ab_sets[a])}, {b} abstains "
                             f"{len(ab_sets[b])}, both {inter}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# grounding-threshold sweep (tuning split only)
# --------------------------------------------------------------------------
def threshold_sweep(meta: dict,
                    thresholds=(0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)):
    seed = int(meta["seed"])
    sample = int(meta["dataset_sample"])
    n_per = int(meta["n_per_stratum"])
    maps = load_dataset(sample, seed)
    idx = index_by_plan(maps)
    queries = [q for q in generate(maps, n_per, seed, report=False)
               if q.split == "tuning"]

    groundable = [q for q in queries if q.stratum in ("direct", "attribute")]
    ungroundable = [q for q in queries if q.stratum in UNGROUNDABLE]

    rows = []
    for t in thresholds:
        # groundable: does the expected node ground its phrase at t?
        answered = sum(
            grounded(idx[q.plan_id], q.expected, q.query_terms[0], t).ok
            for q in groundable)
        # ungroundable: abstain iff no target room grounds the phrase at t.
        far_ab = near_ab = 0
        false_pos = 0        # answered a groundable that should ground? no --
        # abstention precision: of all abstentions, fraction ungroundable.
        abst_total = abst_true = 0
        for q in groundable:
            g = grounded(idx[q.plan_id], q.expected, q.query_terms[0], t).ok
            if not g:
                abst_total += 1          # groundable but would abstain (FP)
        for q in ungroundable:
            rooms = _target_rooms(idx[q.plan_id])
            grounds = any(grounded(idx[q.plan_id], r.id, q.query_terms[0], t).ok
                          for r in rooms)
            if not grounds:
                abst_total += 1
                abst_true += 1
                if q.stratum == "ungroundable_far":
                    far_ab += 1
                else:
                    near_ab += 1
        n_far = sum(q.stratum == "ungroundable_far" for q in ungroundable)
        n_near = sum(q.stratum == "ungroundable_near" for q in ungroundable)
        prec = abst_true / abst_total if abst_total else float("nan")
        rec = abst_true / len(ungroundable) if ungroundable else float("nan")
        rows.append([
            _f(t, 2),
            _f(answered / len(groundable) if groundable else float("nan")),
            _f(far_ab / n_far if n_far else float("nan")),
            _f(near_ab / n_near if n_near else float("nan")),
            _f(prec), _f(rec),
        ])
    headers = ["threshold", "groundable_answered", "far_abstain",
               "near_abstain", "abstain_precision", "abstain_recall"]
    return md_table(headers, rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", help="result CSV(s)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the (slow) tuning-split threshold sweep")
    args = ap.parse_args()

    metas, rows = load_all(args.csvs)
    meta = metas[0]

    print(f"# Ablation analysis\n")
    print(f"- sources: {', '.join(args.csvs)}")
    print(f"- git_commit: {meta.get('git_commit')}  seed: {meta.get('seed')}  "
          f"prompt: {meta.get('prompt_version')}  split: {meta.get('split')}")
    print(f"- rows: {len(rows)}\n")

    print("## Main ablation\n")
    print(table_main(rows) + "\n")

    print(coverage_section(rows))

    models = sorted({r["model"] for r in rows if r["grid"] == "main"})
    enc_tables = [table_sweep(rows, "encoding", "encoding", m) for m in models]
    enc_tables = [t for t in enc_tables if t]
    if enc_tables:
        print("## Encoding sweep (at best main config)\n")
        print("\n\n".join(enc_tables) + "\n")

    k_tables = [table_sweep(rows, "k_hop", "k_hop", m) for m in models]
    k_tables = [t for t in k_tables if t]
    if k_tables:
        print("## k-hop sweep (at best main config)\n")
        print("\n\n".join(k_tables) + "\n")

    if len(models) > 1:
        print("## Model comparison (best main config each)\n")
        print(table_models(rows) + "\n")

    if any(r["stratum"] == "functional" for r in rows):
        print("## Functional stratum\n")
        print(functional_report(rows, meta) + "\n")

    if not args.no_sweep and meta.get("dataset_sample"):
        print("## Grounding-threshold sweep (tuning split, oracle)\n")
        print(threshold_sweep(meta) + "\n")


if __name__ == "__main__":
    main()
