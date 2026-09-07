"""
Smoke test: is a small local model viable for grounded graph queries?

Not a component of the system. A throwaway that answers three questions
before any real code gets written:

  1. Does the model return parseable structured output at all?
  2. Does it resolve correct nodes when the answer exists?
  3. Does it hallucinate when the answer does NOT exist?

Question 3 is the important one. If the model already abstains reliably,
verification adds little and the paper needs rethinking. If it confabulates
freely, verification is where the contribution lives.

Usage:
    python experiments/smoke_test.py
    python experiments/smoke_test.py --model llama3.1:8b
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
# A hand-built graph. Deliberately tiny and unambiguous so that any failure
# is the model's, not the representation's.
# --------------------------------------------------------------------------

GRAPH = {
    "nodes": [
        {"id": "n1", "label": "lobby",       "type": "room",     "pos": [0.0, 0.0]},
        {"id": "n2", "label": "corridor",    "type": "corridor", "pos": [5.0, 0.0]},
        {"id": "n3", "label": "kitchen",     "type": "room",     "pos": [5.0, 4.0]},
        {"id": "n4", "label": "office A",    "type": "room",     "pos": [9.0, 2.0]},
        {"id": "n5", "label": "office B",    "type": "room",     "pos": [9.0, -2.0]},
        {"id": "n6", "label": "charging dock", "type": "dock",   "pos": [1.0, -3.0]},
    ],
    "edges": [
        ["n1", "n2"], ["n2", "n3"], ["n2", "n4"],
        ["n2", "n5"], ["n1", "n6"],
    ],
}

# (query, expected node id or None for ungroundable, stratum)
QUERIES = [
    ("Take this to the kitchen.",                       "n3", "direct"),
    ("Go to office A.",                                 "n4", "direct"),
    ("Deliver to the lobby.",                           "n1", "direct"),
    ("Head to office B please.",                        "n5", "direct"),
    ("Move to the corridor.",                           "n2", "direct"),

    ("Go to the nearest charging dock.",                "n6", "attribute"),
    ("Which room is connected to office A?",            "n2", "relational"),
    ("Take this to the room next to the kitchen.",      "n2", "relational"),
    ("Go to the room where people eat.",                "n3", "attribute"),
    ("Find the place the robot charges.",               "n6", "attribute"),

    ("Take this to the server room.",                   None, "ungroundable"),
    ("Deliver to the third floor conference hall.",     None, "ungroundable"),
    ("Go to the loading bay.",                          None, "ungroundable"),
    ("Bring it to Dr. Rahman's laboratory.",            None, "ungroundable"),
    ("Head to the rooftop garden.",                     None, "ungroundable"),
]

SYSTEM_PROMPT = """You resolve navigation requests against a map of a building.

You are given a map as JSON: a list of nodes (places) and edges (connections).
Given a request, identify which node the robot should navigate to.

Respond with ONLY a JSON object, no other text:
  {"node_id": "<id>", "reason": "<one short sentence>"}

If no node in the map matches the request, respond with:
  {"node_id": null, "reason": "<what is missing>"}

Never invent a node id that is not in the map."""


def build_user_prompt(graph: dict, query: str) -> str:
    return (
        f"Map:\n{json.dumps(graph, indent=None)}\n\n"
        f"The robot is currently at n1 (lobby).\n\n"
        f"Request: {query}"
    )


def parse_response(text: str):
    """Return (parsed_ok, node_id). node_id may be None (a valid answer)."""
    if text is None:
        return False, None
    # Tolerate models that wrap JSON in prose or code fences.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, None
    if "node_id" not in obj:
        return False, None
    return True, obj["node_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")

    valid_ids = {n["id"] for n in GRAPH["nodes"]}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smoke_{args.model.replace(':', '_')}_{stamp}.csv"

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

        parsed_ok, node_id = parse_response(raw)
        correct = parsed_ok and node_id == expected
        # Hallucination: claimed a node when none was correct, or named an id
        # that does not exist in the map at all.
        hallucinated = parsed_ok and (
            (expected is None and node_id is not None)
            or (node_id is not None and node_id not in valid_ids)
        )

        mark = "OK " if correct else "BAD"
        print(f"[{mark}] {stratum:12s} {elapsed:5.1f}s  got={str(node_id):6s} "
              f"want={str(expected):6s}  {query[:44]}")

        rows.append({
            "stratum": stratum,
            "query": query,
            "expected": expected,
            "got": node_id,
            "parsed_ok": parsed_ok,
            "correct": correct,
            "hallucinated": hallucinated,
            "latency_s": round(elapsed, 2),
            "raw": (raw or "").replace("\n", " ")[:300],
        })

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary ----------------------------------------------------------
    n = len(rows)
    ungr = [r for r in rows if r["stratum"] == "ungroundable"]
    grnd = [r for r in rows if r["stratum"] != "ungroundable"]

    print("\n" + "=" * 62)
    print(f"model                : {args.model}")
    print(f"parse rate           : {sum(r['parsed_ok'] for r in rows)}/{n}")
    print(f"accuracy (groundable): {sum(r['correct'] for r in grnd)}/{len(grnd)}")
    print(f"correct abstentions  : {sum(r['correct'] for r in ungr)}/{len(ungr)}")
    print(f"hallucination rate   : {sum(r['hallucinated'] for r in rows)}/{n}")
    print(f"mean latency         : {sum(r['latency_s'] for r in rows) / n:.1f}s")
    print(f"saved                : {out_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()