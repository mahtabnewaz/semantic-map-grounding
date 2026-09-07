#!/usr/bin/env python
"""Ablation runner for the language-grounded navigation study.

Two modes; the pilot must pass before the full grid (CLAUDE.md working
style -- suspect a bug before believing a good number).

  --mode pilot : tuning split only (~199 queries), main grid, one model.
  --mode full  : test split, all grids, all models.

Grids
  main     (n_max, verify_enabled) in [(1,False),(1,True),(2,True),(3,True),
           (5,True)]. verify_enabled=False only pairs with n_max=1; the
           meaningless cells are never generated.
  encoding relational + natural, at the best main config (structured at that
           config is the main-grid baseline, not re-run).
  k_hop    1, 3, and unbounded, at the best main config (k=2 is the main
           baseline, not re-run).
  model    llama3.2:3b, llama3.1:8b (ollama), plus a Groq model guarded by a
           TokenBudget.

Design guarantees
- current_node is drawn once per plan by a seeded RNG over ROOM nodes,
  independent of the query and identical across every config; it is recorded
  in every row. It is never derived from query terms or query.anchor.
- Resumable: rows are appended and flushed incrementally; on restart the
  (query_id, config_id) pairs already present are skipped. A three-hour run
  survives a closed lid.
- The CSV header records git commit, seed, prompt_version, models, provider,
  sample size, split, timestamp, and the full grid. Untraceable rows are
  worthless.
- subgraph_tokens is reported separately from prompt_tokens.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

from data.functional_queries import VERSION as FUNCTIONAL_VERSION  # noqa: E402
from src.agent import AgentConfig, resolve  # noqa: E402
from src.benchmark import generate, index_by_plan  # noqa: E402
from src.graph import NodeType  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402
from src.llm import BudgetExceededError, LLMClient, TokenBudget  # noqa: E402

PROMPT_VERSION = "resolve_v1"
TOKEN_BUDGET = 400
THRESHOLD = 0.60
UNBOUNDED_K = 10 ** 6

OLLAMA_MODELS = ["llama3.2:3b", "llama3.1:8b"]
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODELS = {GROQ_MODEL}

# n_max capped at 3: the pilot showed retries add ~0.5pt for 4x the tokens
# (a deterministic model re-proposes the same answer), so n=5 was dropped.
MAIN_GRID = [(1, False), (1, True), (2, True), (3, True)]
ENCODING_SWEEP = ["relational", "natural"]     # structured comes from main
KHOP_SWEEP = [("1", 1), ("3", 3), ("inf", UNBOUNDED_K)]  # k=2 comes from main

UNGROUNDABLE = {"ungroundable_far", "ungroundable_near"}
STRATA = ["direct", "attribute", "relational",
          "ungroundable_far", "ungroundable_near", "functional"]
_ABBR = {"direct": "d", "attribute": "a", "relational": "r",
         "ungroundable_far": "nf", "ungroundable_near": "nn",
         "functional": "fn"}

COLUMNS = [
    "config_id", "grid", "model", "provider", "n_max", "verify_enabled",
    "encoding", "k_hop", "token_budget", "threshold", "prompt_version",
    "query_id", "plan_id", "stratum", "expected",
    "final_node_id", "abstained", "correct", "attempts",
    "last_failed_predicate", "subgraph_tokens", "prompt_tokens",
    "completion_tokens", "latency_s", "widened", "verify_predicates",
    "status", "current_node",
]


# --------------------------------------------------------------------------
# identity / config helpers
# --------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def query_id(q) -> str:
    import hashlib
    key = f"{q.plan_id}|{q.stratum}|{q.template_id}|{q.text}"
    return "q" + hashlib.sha1(key.encode()).hexdigest()[:12]


def config_id(model: str, grid: str, n_max: int, verify: bool,
              encoding: str, k_label: str) -> str:
    return f"{model}|{grid}|n{n_max}|v{int(verify)}|e{encoding}|k{k_label}"


def main_configs(n_max_filter=None, verify_only=False) -> list[dict]:
    out = []
    for n_max, verify in MAIN_GRID:
        if n_max_filter and n_max not in n_max_filter:
            continue
        if verify_only and not verify:
            continue
        out.append(dict(grid="main", n_max=n_max, verify_enabled=verify,
                        encoding="structured", k_label="2", k_int=2))
    return out


def sweep_configs(best: tuple[int, bool]) -> list[dict]:
    n_max, verify = best
    out = []
    for enc in ENCODING_SWEEP:
        out.append(dict(grid="encoding", n_max=n_max, verify_enabled=verify,
                        encoding=enc, k_label="2", k_int=2))
    for k_label, k_int in KHOP_SWEEP:
        out.append(dict(grid="k_hop", n_max=n_max, verify_enabled=verify,
                        encoding="structured", k_label=k_label, k_int=k_int))
    return out


def build_current_nodes(maps, seed: int) -> dict:
    """One current_node per plan, seeded RNG over ROOM nodes, query-independent."""
    rng = random.Random(seed)
    nodes = {}
    for m in sorted(maps, key=lambda x: x.plan_id):
        rooms = sorted(n.id for n in m.nodes() if n.type is NodeType.ROOM)
        nodes[m.plan_id] = rng.choice(rooms) if rooms else None
    return nodes


def is_correct(stratum: str, expected, final, abstained: bool) -> bool:
    if stratum in UNGROUNDABLE:
        return abstained
    return (not abstained) and final is not None and final == expected


# --------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------
def write_header(f, meta: dict):
    for key in ("git_commit", "timestamp", "mode", "split", "seed",
                "prompt_version", "functional_version", "dataset_sample",
                "loaded_plans", "n_per_stratum", "n_queries", "models",
                "token_budget", "threshold", "main_grid", "encoding_grid",
                "khop_grid"):
        f.write(f"# {key}: {meta[key]}\n")
    csv.writer(f).writerow(COLUMNS)
    f.flush()


def read_meta_and_keys(path: Path):
    """Return (meta dict, set of (query_id, config_id)) from an existing CSV."""
    meta, done = {}, set()
    if not path.exists() or path.stat().st_size == 0:
        return meta, done
    with path.open() as f:
        header_cols = None
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                line = row[0][1:].strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    meta[k] = v
                continue
            if header_cols is None:
                header_cols = row
                continue
            rec = dict(zip(header_cols, row))
            done.add((rec["query_id"], rec["config_id"]))
    return meta, done


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header_cols = None
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if header_cols is None:
                header_cols = row
                continue
            rows.append(dict(zip(header_cols, row)))
    return rows


# --------------------------------------------------------------------------
# best main config (for the sweeps in full mode)
# --------------------------------------------------------------------------
def best_main_config(path: Path, model: str) -> tuple[int, bool]:
    rows = [r for r in read_rows(path)
            if r["grid"] == "main" and r["model"] == model]
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (int(r["n_max"]), r["verify_enabled"] == "True")
        acc[key][1] += 1
        acc[key][0] += 1 if r["correct"] == "True" else 0
    # highest accuracy, tie-break to fewer attempts then to verify on.
    best = max(acc.items(),
               key=lambda kv: (kv[1][0] / kv[1][1] if kv[1][1] else 0,
                               -kv[0][0], kv[0][1]))
    return best[0]


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def run_cells(configs, queries, idx, current_nodes, model, provider, client,
              writer, f, done, pbar, stats, fsync_every=10):
    """Run each (config, query) cell not already present. Returns True if a
    budget refusal stopped this model."""
    written = 0
    for cfg in configs:
        cid = config_id(model, cfg["grid"], cfg["n_max"],
                        cfg["verify_enabled"], cfg["encoding"], cfg["k_label"])
        for q in queries:
            qid = query_id(q)
            if (qid, cid) in done:
                pbar.update(1)
                continue
            cn = current_nodes.get(q.plan_id)
            acfg = AgentConfig(
                n_max=cfg["n_max"], encoding=cfg["encoding"],
                k_hop=cfg["k_int"], token_budget=TOKEN_BUDGET,
                prompt_version=PROMPT_VERSION,
                verify_enabled=cfg["verify_enabled"],
                expand_on_missing=False, current_node=cn, threshold=THRESHOLD)
            try:
                res = resolve(q, idx[q.plan_id], client, acfg)
            except BudgetExceededError as e:
                pbar.write(f"\n[budget] {model}: {e}. Stopping this model.")
                return True
            except Exception as e:                     # keep a long run alive
                pbar.write(f"\n[error] {qid} @ {cid}: "
                           f"{type(e).__name__}: {e}")
                _write_row(writer, cfg, model, provider, q, cn, qid, cid,
                           final=None, abstained=False, correct=False,
                           attempts=0, last_failed=f"error:{type(e).__name__}",
                           sub_tok=0, ptok=0, ctok=0, latency=0.0,
                           widened=False, verify_predicates="",
                           status="invalid")
                done.add((qid, cid))
                pbar.update(1)
                continue

            last_failed = (res.attempt_records[-1].failed_predicate
                           if res.abstained and res.attempt_records else "")
            correct = is_correct(q.stratum, q.expected, res.node_id,
                                 res.abstained)
            _write_row(writer, cfg, model, provider, q, cn, qid, cid,
                       final=res.node_id, abstained=res.abstained,
                       correct=correct, attempts=res.attempts,
                       last_failed=last_failed or "",
                       sub_tok=res.subgraph_tokens, ptok=res.prompt_tokens,
                       ctok=res.completion_tokens, latency=res.latency_s,
                       widened=res.widened,
                       verify_predicates=res.verify_predicates,
                       status=res.status)
            done.add((qid, cid))

            # invalid rows (transport error / empty reply) are excluded from
            # running accuracy -- they are infrastructure failures, not answers.
            if res.status != "invalid":
                s = stats[q.stratum]
                s[1] += 1
                s[0] += 1 if correct else 0
            written += 1
            f.flush()
            if written % fsync_every == 0:
                os.fsync(f.fileno())
            pbar.set_postfix_str(_running_str(stats))
            pbar.update(1)
    os.fsync(f.fileno())
    return False


def _write_row(writer, cfg, model, provider, q, cn, qid, cid, *, final,
               abstained, correct, attempts, last_failed, sub_tok, ptok,
               ctok, latency, widened, verify_predicates, status):
    writer.writerow([
        cid, cfg["grid"], model, provider, cfg["n_max"],
        cfg["verify_enabled"], cfg["encoding"], cfg["k_label"], TOKEN_BUDGET,
        THRESHOLD, PROMPT_VERSION, qid, q.plan_id, q.stratum,
        q.expected if q.expected is not None else "",
        final if final is not None else "", abstained, correct, attempts,
        last_failed, sub_tok, ptok, ctok, f"{latency:.4f}", widened,
        verify_predicates, status, cn or "",
    ])


def _running_str(stats) -> str:
    parts = []
    for s in STRATA:
        c, t = stats[s]
        if t:
            parts.append(f"{_ABBR[s]}={c / t:.2f}")
    return " ".join(parts)


def make_client(model: str, provider: str, budget, max_tokens=200,
                reasoning_effort=None):
    if provider == "groq":
        return LLMClient("groq", model, budget=budget, max_tokens=max_tokens,
                         reasoning_effort=reasoning_effort)
    return LLMClient("ollama", model, max_tokens=max_tokens)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def final_report(path: Path):
    rows = read_rows(path)
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config_id"]].append(r)

    print("\n" + "=" * 78)
    print("FINAL REPORT")
    print("=" * 78)

    # Verification coverage per stratum (which predicates were applicable).
    # Reported so accuracy is never averaged across strata with different
    # verification available -- functional has no semantic predicate.
    cov = defaultdict(set)
    for r in rows:
        if r.get("verify_predicates"):
            cov[r["stratum"]].add(r["verify_predicates"])
    if cov:
        print("\nverification coverage (applicable predicates per stratum):")
        for s in STRATA:
            if cov.get(s):
                print(f"  {s:18s}: {' | '.join(sorted(cov[s]))}")

    for cid in sorted(by_cfg):
        all_rs = by_cfg[cid]
        invalid = [r for r in all_rs if r.get("status") == "invalid"]
        rs = [r for r in all_rs if r.get("status") != "invalid"]
        note = f"  [EXCLUDED {len(invalid)} invalid rows]" if invalid else ""
        print(f"\n{cid}   (n={len(rs)} scored){note}")
        # per-stratum accuracy (invalid rows already excluded)
        for s in STRATA:
            sub = [r for r in rs if r["stratum"] == s]
            if not sub:
                continue
            acc = sum(r["correct"] == "True" for r in sub) / len(sub)
            print(f"  acc[{s:17s}] = {acc:.3f}  (n={len(sub)})")
        # abstention precision / recall on the ungroundable strata
        abstained = [r for r in rs if r["abstained"] == "True"]
        true_abstain = [r for r in abstained if r["stratum"] in UNGROUNDABLE]
        prec = len(true_abstain) / len(abstained) if abstained else float("nan")
        for s in ("ungroundable_far", "ungroundable_near"):
            sub = [r for r in rs if r["stratum"] == s]
            rec = (sum(r["abstained"] == "True" for r in sub) / len(sub)
                   if sub else float("nan"))
            print(f"  abstain recall[{s:17s}] = {rec:.3f}")
        print(f"  abstain precision (all strata) = {prec:.3f}")
        att = [int(r["attempts"]) for r in rs if r["attempts"].isdigit()]
        lat = [float(r["latency_s"]) for r in rs]
        stok = [int(r["subgraph_tokens"]) for r in rs
                if r["subgraph_tokens"].isdigit()]
        ptok = [int(r["prompt_tokens"]) for r in rs
                if r["prompt_tokens"].isdigit()]
        print(f"  mean attempts={sum(att) / len(att):.2f}  "
              f"mean latency={sum(lat) / len(lat):.2f}s  "
              f"mean subgraph_tokens={sum(stok) / len(stok):.0f}  "
              f"mean prompt_tokens={sum(ptok) / len(ptok):.0f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=400,
                    help="HouseExpo plans to load")
    ap.add_argument("--n-per-stratum", type=int, default=200)
    ap.add_argument("--pilot-model", default="llama3.2:3b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #queries (smoke/validation only)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model override")
    ap.add_argument("--split", choices=["tuning", "test"], default=None,
                    help="override the split")
    ap.add_argument("--n-max", default=None,
                    help="comma-separated n_max values to keep from the main "
                         "grid (e.g. '1')")
    ap.add_argument("--no-sweeps", action="store_true",
                    help="main grid only; skip encoding/k-hop sweeps")
    ap.add_argument("--verify-only", action="store_true",
                    help="keep only verify_enabled=True configs")
    ap.add_argument("--provider", choices=["ollama", "groq"], default=None,
                    help="force provider for --models (else inferred)")
    ap.add_argument("--budget-file", default=None,
                    help="token-budget JSON path (give concurrent runs "
                         "separate files to avoid races)")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="completion token budget (raise for reasoning models)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning models: low/medium/high (e.g. gpt-oss)")
    ap.add_argument("--strata", default=None,
                    help="comma-separated strata to keep (e.g. 'functional')")
    ap.add_argument("--groq-cap", type=int, default=None,
                    help="override daily groq token cap (paid models: set high)")
    args = ap.parse_args()

    pilot = args.mode == "pilot"
    split = args.split or ("tuning" if pilot else "test")
    n_max_filter = ({int(x) for x in args.n_max.split(",")}
                    if args.n_max else None)
    no_sweeps = args.no_sweeps or pilot
    out = Path(args.out) if args.out else (
        ROOT / "experiments" / "results" /
        f"ablation_{args.mode}_seed{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.sample} plans, seed {args.seed} ...")
    maps = load_dataset(args.sample, args.seed)
    idx = index_by_plan(maps)
    all_q = generate(maps, args.n_per_stratum, args.seed, report=False)
    keep = set(args.strata.split(",")) if args.strata else None
    queries = [q for q in all_q if q.split == split
               and (keep is None or q.stratum in keep)]
    queries.sort(key=query_id)
    if args.limit:
        queries = queries[:args.limit]
    current_nodes = build_current_nodes(maps, args.seed)
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = [args.pilot_model] if pilot else OLLAMA_MODELS + [GROQ_MODEL]

    meta = {
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode, "split": split, "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "functional_version": FUNCTIONAL_VERSION, "dataset_sample": args.sample,
        "loaded_plans": len(maps), "n_per_stratum": args.n_per_stratum,
        "n_queries": len(queries), "models": ",".join(models),
        "token_budget": TOKEN_BUDGET, "threshold": THRESHOLD,
        "main_grid": ";".join(f"({n},{v})" for n, v in MAIN_GRID),
        "encoding_grid": "structured(main);" + ";".join(ENCODING_SWEEP),
        "khop_grid": "2(main);" + ";".join(k for k, _ in KHOP_SWEEP),
    }

    prev_meta, done = read_meta_and_keys(out)
    if prev_meta:
        for k in ("seed", "split", "prompt_version"):
            if str(prev_meta.get(k)) != str(meta[k]):
                sys.exit(f"[abort] {out} was written with {k}="
                         f"{prev_meta.get(k)!r}, not {meta[k]!r}. "
                         f"Use a different --out.")
        print(f"[resume] {len(done)} (query,config) cells already present")

    # total planned cells for the progress bar
    n_main = len(main_configs(n_max_filter, args.verify_only)) * len(queries)
    n_sweep = 0 if no_sweeps else len(sweep_configs((1, True))) * len(queries)
    total_cells = len(models) * (n_main + n_sweep)

    budget_path = (Path(args.budget_file) if args.budget_file
                   else out.parent / "token_budget.json")
    budget = TokenBudget(budget_path)
    # Daily Groq cap scales with the number of keys available (failover).
    if args.groq_cap is not None:
        budget.caps["groq"] = args.groq_cap
    else:
        try:
            from src.llm import groq_api_keys
            budget.caps["groq"] = 100_000 * len(groq_api_keys())
        except Exception:
            pass
    new_file = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            write_header(f, meta)
        stats = defaultdict(lambda: [0, 0])
        with tqdm(total=total_cells, desc=f"{args.mode}", unit="cell") as pbar:
            for model in models:
                provider = args.provider or (
                    "groq" if model in GROQ_MODELS else "ollama")
                client = make_client(model, provider, budget,
                                     max_tokens=args.max_tokens,
                                     reasoning_effort=args.reasoning_effort)
                stopped = run_cells(main_configs(n_max_filter, args.verify_only), queries, idx,
                                    current_nodes, model, provider, client,
                                    writer, f, done, pbar, stats)
                if stopped or no_sweeps:
                    continue
                best = best_main_config(out, model)
                run_cells(sweep_configs(best), queries, idx, current_nodes,
                          model, provider, client, writer, f, done, pbar,
                          stats)

    final_report(out)
    print(f"\n[done] rows in {out}")


if __name__ == "__main__":
    main()
