#!/usr/bin/env python
"""Generate the paper's figures from result CSVs.

Reads a manifest of {method: (templated_csv, functional_csv)} and produces:
  fig_flip      grouped bars: templated vs functional accuracy per method
                (the central result: the matcher and the models swap places)
  fig_scale     accuracy vs model size (3B/27B/120B), templated + functional
  fig_percat    per-category functional accuracy vs lexical groundability

Invalid rows (status=invalid: transport error / empty reply) are excluded via
analyse.metrics. Colours are the Okabe--Ito colour-blind-safe palette; figures
are vector PDFs sized for a single IEEE column. Re-run after any rerun.

Usage: figures.py {tuning|test}
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.functional_queries import FUNCTIONAL_QUERIES  # noqa: E402
from experiments.analyse import _acc, read_csv  # noqa: E402
from experiments.run_ablation import query_id  # noqa: E402
from src.benchmark import _target_rooms, generate, index_by_plan  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402
from src.verify import _grounding_score  # noqa: E402

FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Okabe--Ito, colour-blind safe and legible in greyscale.
COLOR = {"lexical": "#999999", "Llama-3B": "#E69F00",
         "Qwen-27B": "#56B4E9", "GPT-OSS-120B": "#009E73"}
SIZE = {"Llama-3B": 3, "Qwen-27B": 27, "GPT-OSS-120B": 120}

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _valid(rows):
    return [r for r in rows if r.get("status") != "invalid"]


def _overall(csv, n_max=None, verify=None):
    """Overall accuracy over valid rows, optionally filtered by config."""
    if csv is None or not Path(csv).exists():
        return None, 0
    _, rows = read_csv(Path(csv))
    inv = sum(r.get("status") == "invalid" for r in rows)
    rows = _valid(rows)
    if n_max is not None:
        rows = [r for r in rows if int(r["n_max"]) == n_max]
    if verify is not None:
        rows = [r for r in rows if (r["verify_enabled"] == "True") == verify]
    return (_acc(rows) if rows else None), inv


def manifest(split):
    r = ROOT / "experiments" / "results"
    def t(m): return r / f"{'test_' if split=='test' else ''}templated_{m}_seed0.csv"
    def f(m): return r / f"{'test_' if split=='test' else ''}functional_{m}_seed0.csv"
    # tuning templated CSVs were produced under different filenames
    if split == "tuning":
        tmpl = {"lexical": r / "baseline_lexical_seed0.csv",
                "Llama-3B": r / "ablation_pilot_v2_seed0.csv",
                "Qwen-27B": r / "model_scale_qwen_seed0.csv",
                "GPT-OSS-120B": r / "model_scale_gpt-oss-120b_seed0.csv"}
        fun = {"lexical": r / "functional_lexical_seed0.csv",
               "Llama-3B": r / "functional_3b_seed0.csv",
               "Qwen-27B": r / "functional_qwen_seed0.csv",
               "GPT-OSS-120B": r / "functional_gptoss120b_seed0.csv"}
    else:
        tmpl = {"lexical": t("lexical"), "Llama-3B": t("3b"),
                "Qwen-27B": t("qwen"), "GPT-OSS-120B": t("gptoss120b")}
        fun = {"lexical": f("lexical"), "Llama-3B": f("3b"),
               "Qwen-27B": f("qwen"), "GPT-OSS-120B": f("gptoss120b")}
    return tmpl, fun


def fig_flip(tmpl, fun, split):
    methods = ["lexical", "Llama-3B", "Qwen-27B", "GPT-OSS-120B"]
    tvals = [_overall(tmpl[m])[0] if m == "lexical"
             else _overall(tmpl[m], n_max=1, verify=True)[0] for m in methods]
    fvals = [_overall(fun[m])[0] for m in methods]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = range(len(methods))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [v or 0 for v in tvals], w,
           label="templated", color="#4477AA", edgecolor="black", linewidth=0.4)
    ax.bar([i + w / 2 for i in x], [v or 0 for v in fvals], w,
           label="functional", color="#EE6677", edgecolor="black",
           linewidth=0.4, hatch="////")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax.set_title(f"Templated vs. functional ({split} split)", fontsize=8)
    out = FIGDIR / f"fig_flip_{split}.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  {out.name}  templated={[round(v,3) if v else None for v in tvals]}"
          f"  functional={[round(v,3) if v else None for v in fvals]}")


def fig_scale(tmpl, fun, split):
    models = ["Llama-3B", "Qwen-27B", "GPT-OSS-120B"]
    xs = [SIZE[m] for m in models]
    t = [_overall(tmpl[m], n_max=1, verify=True)[0] for m in models]
    f = [_overall(fun[m])[0] for m in models]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(xs, [v or 0 for v in t], "o-", color="#4477AA", label="templated")
    ax.plot(xs, [v or 0 for v in f], "s--", color="#EE6677", label="functional")
    # lexical reference lines
    lt = _overall(tmpl["lexical"])[0]
    lf = _overall(fun["lexical"])[0]
    if lt is not None:
        ax.axhline(lt, color="#999999", lw=0.8, ls=":")
        ax.text(xs[-1], lt + 0.01, "lexical (templated)", fontsize=6,
                ha="right", color="#666666")
    if lf is not None:
        ax.axhline(lf, color="#999999", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}B" for s in xs])
    ax.set_xlabel("model size (parameters, log scale)")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Accuracy vs. model size ({split} split)", fontsize=8)
    out = FIGDIR / f"fig_scale_{split}.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  {out.name}  templated={[round(v,3) if v else None for v in t]}"
          f"  functional={[round(v,3) if v else None for v in f]}")


def fig_percat(fun, split, meta_csv):
    _, rows0 = read_csv(Path(meta_csv))
    meta, _ = read_csv(Path(meta_csv))
    seed = int(meta.get("seed", 0)); sample = int(meta.get("dataset_sample", 400))
    n_per = int(meta.get("n_per_stratum", 800))
    fsplit = meta.get("split", split)
    maps = load_dataset(sample, seed)
    fqs = [q for q in generate(maps, n_per, seed, report=False)
           if q.stratum == "functional" and q.split == fsplit]
    qid_cat = {query_id(q): q.template_id.split(".")[1].split("#")[0] for q in fqs}
    rep = {}
    for m in maps:
        for n in _target_rooms(m):
            rep.setdefault(n.label, (m, n.id))
    ground = {c: (statistics.mean([_grounding_score(*rep[c], p) for p in ph])
                  if c in rep else 0)
              for c, ph in FUNCTIONAL_QUERIES.items()}

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in ["Llama-3B", "Qwen-27B", "GPT-OSS-120B"]:
        if not Path(fun[m]).exists():
            continue
        _, rows = read_csv(Path(fun[m]))
        rows = _valid(rows)
        cats = sorted(set(qid_cat.values()))
        xs, ys = [], []
        for c in cats:
            sub = [r for r in rows if qid_cat.get(r["query_id"]) == c]
            if sub:
                xs.append(ground[c])
                ys.append(sum(r["correct"] == "True" for r in sub) / len(sub))
        ax.scatter(xs, ys, s=18, color=COLOR[m], label=m, edgecolor="black",
                   linewidth=0.3, alpha=0.85)
    ax.set_xlabel("per-category lexical groundability")
    ax.set_ylabel("functional accuracy"); ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title(f"Model accuracy vs. groundability ({split})", fontsize=8)
    out = FIGDIR / f"fig_percat_{split}.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  {out.name}")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    tmpl, fun = manifest(split)
    print(f"[figures] split={split} -> {FIGDIR}")
    fig_flip(tmpl, fun, split)
    fig_scale(tmpl, fun, split)
    fig_percat(fun, split, fun["Llama-3B"])


if __name__ == "__main__":
    main()
