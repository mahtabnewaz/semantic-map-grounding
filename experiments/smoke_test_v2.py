"""
Smoke test v2: can a deterministic grounding check catch semantic snapping?

v1 finding: models do not invent node ids. They snap an ungroundable
request to the semantically nearest existing node ("server room" -> office A).
Existence and reachability checks pass such answers, so the verification
predicate specified in CLAUDE.md is insufficient as written.

Hypothesis under test: if nodes carry semantic attributes, a purely lexical
grounding check can separate

    "the room where people eat"  -> kitchen        (legitimate, low label overlap)
    "the server room"            -> office A       (snapped, low label overlap)

by matching against attributes rather than labels alone. No embedding model,
no model judging a model. Invariant 2 holds.

Outputs a threshold sweep so the operating point is chosen from data.

Usage:
    python experiments/smoke_test_v2.py
    python experiments/smoke_test_v2.py --model llama3.1:8b
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# --------------------------------------------------------------------------
# Same topology as v1, now with semantic attributes on every node.
# Attributes are what the operator would plausibly type when annotating.
# --------------------------------------------------------------------------

GRAPH = {
    "nodes": [
        {"id": "n1", "label": "lobby", "type": "room", "pos": [0.0, 0.0],
         "attributes": ["entrance", "reception", "waiting", "front desk"]},
        {"id": "n2", "label": "corridor", "type": "corridor", "pos": [5.0, 0.0],
         "attributes": ["hallway", "passage", "walkway"]},
        {"id": "n3", "label": "kitchen", "type": "room", "pos": [5.0, 4.0],
         "attributes": ["food", "eating", "dining", "meals", "pantry", "cafeteria"]},
        {"id": "n4", "label": "office A", "type": "room", "pos": [9.0, 2.0],
         "attributes": ["desk", "work", "workstation", "staff"]},
        {"id": "n5", "label": "office B", "type": "room", "pos": [9.0, -2.0],
         "attributes": ["desk", "work", "workstation", "staff"]},
        {"id": "n6", "label": "charging dock", "type": "dock", "pos": [1.0, -3.0],
         "attributes": ["charge", "charging", "power", "battery", "recharge"]},
    ],
    "edges": [
        ["n1", "n2"], ["n2", "n3"], ["n2", "n4"],
        ["n2", "n5"], ["n1", "n6"],
    ],
}

# (query, expected node id or None, stratum)
# The ungroundable stratum now includes near-misses: plausible names that
# almost fit the map. These are the hard cases.
QUERIES = [
    ("Take this to the kitchen.",                    "n3", "direct"),
    ("Go to office A.",                              "n4", "direct"),
    ("Deliver to the lobby.",                        "n1", "direct"),
    ("Head to office B please.",                     "n5", "direct"),
    ("Move to the corridor.",                        "n2", "direct"),

    ("Go to the nearest charging dock.",             "n6", "attribute"),
    ("Go to the room where people eat.",             "n3", "attribute"),
    ("Find the place the robot charges.",            "n6", "attribute"),
    ("Take it to the reception area.",               "n1", "attribute"),
    ("Deliver to a staff workstation.",              "n4", "attribute"),

    ("Which place is connected to office A?",        "n2", "relational"),
    ("Go to the place that connects to the kitchen.", "n2", "relational"),
    ("Take this to the room across from office A.",  "n5", "relational"),

    # Far misses: nothing remotely similar in the map.
    ("Take this to the server room.",                None, "ungroundable_far"),
    ("Head to the rooftop garden.",                  None, "ungroundable_far"),
    ("Deliver to the third floor conference hall.",  None, "ungroundable_far"),

    # Near misses: plausible given the map, but absent. Hardest stratum.
    ("Take this to office C.",                       None, "ungroundable_near"),
    ("Go to the north corridor.",                    None, "ungroundable_near"),
    ("Deliver to the second kitchen.",               None, "ungroundable_near"),
    ("Head to the staff lounge.",                    None, "ungroundable_near"),
]

SYSTEM_PROMPT = """You resolve navigation requests against a map of a building.

You are given a map as JSON: nodes (places, each with a label and semantic
attributes) and edges (connections between them).

Given a request, identify which node the robot should navigate to.

Respond with ONLY a JSON object, no other text:
  {"target_phrase": "<the destination phrase from the request>",
   "node_id": "<id>",
   "node_label": "<that node's label>"}

If no node in the map matches the request, respond with:
  {"target_phrase": "<the destination phrase from the request>",
   "node_id": null,
   "node_label": null}

Never invent a node id or label that is not in the map."""

STOPWORDS = {
    "the", "a", "an", "to", "this", "that", "it", "please", "go", "take",
    "deliver", "head", "move", "find", "bring", "room", "place", "area",
    "of", "for", "and", "is", "where", "which", "there", "at", "in", "on",
}


def build_user_prompt(graph: dict, query: str) -> str:
    return (
        f"Map:\n{json.dumps(graph)}\n\n"
        f"The robot is currently at n1 (lobby).\n\n"
        f"Request: {query}"
    )


def tokens(text: str) -> set:
    """Content words, lowercased, stopwords removed."""
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def grounding_score(target_phrase: str, node: dict) -> float:
    """
    Deterministic. Fraction of the target phrase's content words that appear
    in the node's label or attributes. No model involved.

    "where people eat" vs kitchen{food,eating,dining} -> 'eat' stems near
    'eating', so we also allow prefix matching on words of length >= 4.
    """
    target = tokens(target_phrase)
    if not target:
        return 0.0

    vocab = tokens(node["label"]) | tokens(" ".join(node.get("attributes", [])))

    hits = 0
    for t in target:
        if t in vocab:
            hits += 1
            continue
        # crude stemming: prefix match both directions, min length 4
        if len(t) >= 4 and any(
            v.startswith(t[:4]) or t.startswith(v[:4]) for v in vocab if len(v) >= 4
        ):
            hits += 1
    return hits / len(target)


def parse_response(text: str):
    if text is None:
        return False, None, None, None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False, None, None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, None, None, None
    if "node_id" not in obj:
        return False, None, None, None
    return True, obj.get("node_id"), obj.get("node_label"), obj.get("target_phrase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")
    nodes_by_id = {n["id"]: n for n in GRAPH["nodes"]}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smoke2_{args.model.replace(':', '_')}_{stamp}.csv"

    rows = []
    print(f"model: {args.model}\n")

    for query, expected, stratum in QUERIES:
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(GRAPH, query)},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content
        except Exception as exc:                      # noqa: BLE001
            raw = None
            print(f"  API error: {exc}")
        elapsed = time.time() - t0

        ok, node_id, node_label, target_phrase = parse_response(raw)

        # Structural checks (the v1 verifier)
        exists = ok and node_id in nodes_by_id if node_id else None
        label_consistent = (
            ok and node_id in nodes_by_id
            and (node_label or "").strip().lower()
            == nodes_by_id[node_id]["label"].lower()
        ) if node_id else None

        # New: grounding score against label + attributes
        score = (
            grounding_score(target_phrase, nodes_by_id[node_id])
            if node_id and node_id in nodes_by_id else None
        )

        raw_correct = ok and node_id == expected
        mark = "OK " if raw_correct else "BAD"
        score_s = f"{score:.2f}" if score is not None else "  - "
        print(f"[{mark}] {stratum:18s} {elapsed:5.1f}s  got={str(node_id):5s} "
              f"want={str(expected):5s}  ground={score_s}  {query[:38]}")

        rows.append({
            "stratum": stratum,
            "query": query,
            "target_phrase": target_phrase,
            "expected": expected,
            "got": node_id,
            "node_label": node_label,
            "parsed_ok": ok,
            "exists": exists,
            "label_consistent": label_consistent,
            "grounding_score": score,
            "raw_correct": raw_correct,
            "latency_s": round(elapsed, 2),
        })

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- threshold sweep --------------------------------------------------
    # At threshold T, reject any answer whose grounding score < T (treat as
    # abstention). Report resulting accuracy.
    print("\nGrounding threshold sweep")
    print(f"{'T':>5}  {'correct':>8}  {'wrongly rejected':>17}  {'halluc. caught':>15}")
    for t in [0.0, 0.2, 0.34, 0.4, 0.5, 0.6, 0.75, 1.0]:
        correct = wrongly_rejected = caught = 0
        for r in rows:
            if not r["parsed_ok"]:
                continue
            s = r["grounding_score"]
            rejected = (s is not None and s < t)
            final = None if (r["got"] is None or rejected) else r["got"]
            if final == r["expected"]:
                correct += 1
            if rejected and r["got"] == r["expected"]:
                wrongly_rejected += 1
            if rejected and r["expected"] is None and r["got"] is not None:
                caught += 1
        print(f"{t:5.2f}  {correct:8d}  {wrongly_rejected:17d}  {caught:15d}")

    n = len(rows)
    far = [r for r in rows if r["stratum"] == "ungroundable_far"]
    near = [r for r in rows if r["stratum"] == "ungroundable_near"]
    print("\n" + "=" * 66)
    print(f"model                     : {args.model}")
    print(f"parse rate                : {sum(r['parsed_ok'] for r in rows)}/{n}")
    print(f"raw accuracy (no verify)  : {sum(r['raw_correct'] for r in rows)}/{n}")
    print(f"abstained, far misses     : {sum(r['got'] is None for r in far)}/{len(far)}")
    print(f"abstained, near misses    : {sum(r['got'] is None for r in near)}/{len(near)}")
    print(f"mean latency              : {sum(r['latency_s'] for r in rows)/n:.1f}s")
    print(f"saved                     : {out_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()