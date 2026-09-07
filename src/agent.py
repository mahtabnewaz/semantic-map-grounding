"""The plan-retrieve-verify cycle.

resolve(query, semantic_map, client, config) runs, per CLAUDE.md:

 1. retrieve  -- scope a subgraph with serialize.scoped_subgraph, anchored
                 at the robot's current node, using the query's query_terms
 2. serialise -- under the configured encoding
 3. call      -- the LLM with the configured prompt version
 4. parse     -- unparseable output counts as a failed attempt
 5. verify    -- verify.verify(), dispatched on the query stratum
 6. retry     -- on failure, append the diagnostic to context and retry
 7. abstain   -- after n_max attempts

Invariants (CLAUDE.md):
- The LLM returns a node id and, if relational, a symbolic relation -- never
  a path or coordinates (#1). The prompt enforces this; the agent never asks
  for metric output.
- Verification is a pure function of (query, proposed node id, map) (#2). The
  model's only influence on the predicate is which node it names: grounding
  scores the QUERY's phrase (query_terms), and relation checks use the
  QUERY's relation and anchor -- never the model's restatement of them. The
  model's claimed target_phrase/relation/anchor are recorded on each attempt
  for diagnostics but never enter verify().
- Abstention is a first-class outcome, never an error (#5).

Ablation flags (verify_enabled, expand_on_missing) are genuinely
independent of n_max and of each other: verify_enabled=False skips
verification and accepts the first parsed answer; n_max=1 with
verify_enabled=False is the single-pass baseline. Parsing always runs, so an
unparseable answer is a failed attempt under any flag combination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .graph import SemanticMap
from .prompts import get_prompt
from .serialize import ENCODERS, scoped_subgraph
from .verify import (
    VerificationResult,
    _content_stems,
    _node_stem_set,
    verify as verify_proposal,
)

# Query stratum -> verify() query_type. Ungroundable strata are checked as
# direct grounding: nothing should ground, so they fail and abstain.
_VERIFY_TYPE = {
    "direct": "direct",
    "attribute": "attribute",
    "relational": "relational",
    "ungroundable_far": "direct",
    "ungroundable_near": "direct",
    "functional": "functional",
}

# Semantic + structural predicates each query_type is verified against, for
# per-row coverage reporting. Functional has NO semantic predicate (grounding
# has no lexical route and a category check would be an oracle), so its
# verification coverage is reduced -- recorded so the analysis never averages
# it in with strata that carry a semantic predicate.
_APPLICABLE_PREDICATES = {
    "direct": "node_exists,grounded,reachable",
    "attribute": "node_exists,grounded,reachable",
    "relational": "node_exists,relation_holds,reachable",
    "functional": "node_exists,reachable",
}


@dataclass
class AgentConfig:
    """Configuration + ablation switches for one resolve run."""

    n_max: int = 3
    encoding: str = "structured"
    k_hop: int = 2
    token_budget: int = 400
    prompt_version: str = "resolve_v1"
    verify_enabled: bool = True
    expand_on_missing: bool = False
    # The robot's current node (retrieval anchor + reachability start). If
    # unset, a query-relevant node is chosen deterministically.
    current_node: Optional[str] = None
    threshold: float = 0.60


@dataclass
class AttemptRecord:
    """Everything one cycle iteration produced, for the results table."""

    attempt: int
    raw_text: str
    parsed: bool
    proposal: Optional[dict]           # what was verified (query-derived)
    model_output: Optional[dict]       # the model's raw claims, for diagnostics
    verified: Optional[bool]           # None if verification was skipped
    failed_predicate: Optional[str]
    diagnostic: Optional[str]
    widened: bool
    subgraph_size: int
    dropped: list
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    llm_error: Optional[str]


@dataclass
class AgentResult:
    """Outcome of resolve(), traceable to the prompt and encoding used."""

    node_id: Optional[str]             # None on abstention
    abstained: bool
    attempts: int
    attempt_records: list[AttemptRecord]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    encoding: str
    prompt_version: str
    subgraph_size: int
    subgraph_tokens: int
    dropped: list
    widened: bool
    verify_predicates: str            # coverage: predicates this query_type uses
    status: str                       # "ok", or "invalid" (no real LLM response)
    stratum: str
    plan_id: Optional[str]
    query_text: str

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["attempt_records"] = [dict(r.__dict__) for r in self.attempt_records]
        return d


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve_anchor(m: SemanticMap, config: AgentConfig, query) -> str:
    """The robot's current node if given, else the node best matching the
    query terms (deterministic, id-tie-broken)."""
    if config.current_node and m.has_node(config.current_node):
        return config.current_node
    q_stems: set[str] = set()
    for term in query.query_terms:
        q_stems.update(_content_stems(term))
    best_id = None
    best_score = -1
    for node in m.nodes():
        score = len(q_stems & _node_stem_set(m, node.id))
        if score > best_score or (score == best_score
                                  and (best_id is None or node.id < best_id)):
            best_id, best_score = node.id, score
    if best_id is None:
        raise ValueError("cannot resolve a retrieval anchor in an empty map")
    return best_id


def _parse(text: str) -> Optional[dict]:
    """Extract the JSON object from model output, or None if unparseable.

    Content validity (right keys, sensible values) is NOT judged here -- a
    parsed dict with a bad node_id is a verification failure, not a parse
    failure. Only genuinely non-JSON output counts as unparseable."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate[:4].lower() == "json":
            candidate = candidate[4:]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    i, j = candidate.find("{"), candidate.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(candidate[i:j + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def resolve(query, semantic_map: SemanticMap, client,
            config: AgentConfig) -> AgentResult:
    """Resolve a query to a node id, or abstain. See module docstring."""
    prompt = get_prompt(config.prompt_version)
    anchor = _resolve_anchor(semantic_map, config, query)
    query_type = _VERIFY_TYPE.get(query.stratum, "direct")
    verify_predicates = (
        _APPLICABLE_PREDICATES.get(query_type, "node_exists,reachable")
        if config.verify_enabled else "disabled")

    current_k = config.k_hop
    records: list[AttemptRecord] = []
    feedback_lines: list[str] = []
    cum_ptok = cum_ctok = 0
    total_latency = 0.0
    final_node: Optional[str] = None
    abstained = True
    widened_ever = False
    last_scope = None
    got_real_response = False   # any attempt with a non-error, non-empty reply

    for attempt in range(1, config.n_max + 1):
        scope = scoped_subgraph(semantic_map, anchor, query.query_terms,
                                current_k, config.token_budget, config.encoding)
        last_scope = scope
        subgraph_text = ENCODERS[config.encoding](
            semantic_map, scope.node_ids, anchor)
        feedback = "\n".join(feedback_lines) if feedback_lines else "(none)"
        user = prompt.render(anchor_id=anchor, encoding=config.encoding,
                             subgraph=subgraph_text, query=query.text,
                             feedback=feedback)

        resp = client.complete(prompt.system, user)
        cum_ptok += resp.prompt_tokens
        cum_ctok += resp.completion_tokens
        total_latency += resp.latency_s
        # A usable response has no transport error and non-empty content.
        # A transport error or empty completion is an infrastructure failure
        # (rate limit, reasoning-truncation), NOT a model answer -- see status.
        if resp.error is None and resp.text.strip():
            got_real_response = True

        parsed = _parse(resp.text) if resp.error is None else None
        if parsed is None:
            diag = resp.error or "model output was not parseable JSON"
            records.append(AttemptRecord(
                attempt=attempt, raw_text=resp.text, parsed=False,
                proposal=None, model_output=None, verified=None,
                failed_predicate="parse", diagnostic=diag, widened=False,
                subgraph_size=len(scope.node_ids), dropped=list(scope.dropped),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                latency_s=resp.latency_s, llm_error=resp.error))
            feedback_lines.append(
                "Previous output was not valid JSON with the required keys; "
                "return only the JSON object.")
            continue

        # The model's raw claims -- recorded for diagnostics, NEVER fed into
        # the verification predicate (invariant 2: no model output judges the
        # model). Only the chosen node_id crosses into the proposal.
        model_output = {
            "node_id": parsed.get("node_id"),
            "target_phrase": parsed.get("target_phrase"),
            "relation": parsed.get("relation"),
            "anchor_id": parsed.get("anchor_id"),
        }
        # Verification uses the QUERY's phrase/relation/anchor, not the
        # model's restatement of them.
        query_phrase = (" ".join(query.query_terms) if query.query_terms
                        else query.text)
        proposal = {
            "node_id": model_output["node_id"],
            "target_phrase": query_phrase,
            "anchor_id": getattr(query, "anchor", None),
            "relation": getattr(query, "relation", None),
            "from_id": anchor,
        }

        if not config.verify_enabled:
            # Ablation: accept the first parsed answer, no verification.
            final_node = model_output["node_id"]
            abstained = False
            records.append(AttemptRecord(
                attempt=attempt, raw_text=resp.text, parsed=True,
                proposal=proposal, model_output=model_output, verified=None,
                failed_predicate=None, diagnostic=None, widened=False,
                subgraph_size=len(scope.node_ids), dropped=list(scope.dropped),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                latency_s=resp.latency_s, llm_error=None))
            break

        vres: VerificationResult = verify_proposal(
            semantic_map, proposal, query_type, threshold=config.threshold)

        widened_this = False
        if not vres.ok and config.expand_on_missing \
                and vres.failed_predicate == "grounded":
            current_k += 1        # widen for the next attempt
            widened_this = widened_ever = True

        records.append(AttemptRecord(
            attempt=attempt, raw_text=resp.text, parsed=True,
            proposal=proposal, model_output=model_output, verified=vres.ok,
            failed_predicate=vres.failed_predicate, diagnostic=vres.diagnostic,
            widened=widened_this, subgraph_size=len(scope.node_ids),
            dropped=list(scope.dropped), prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_s=resp.latency_s, llm_error=None))

        if vres.ok:
            final_node = proposal["node_id"]
            abstained = False
            break

        feedback_lines.append(
            f"Answer '{proposal['node_id']}' was rejected: {vres.diagnostic}")

    return AgentResult(
        node_id=final_node, abstained=abstained, attempts=len(records),
        attempt_records=records, prompt_tokens=cum_ptok,
        completion_tokens=cum_ctok, latency_s=total_latency,
        encoding=config.encoding, prompt_version=config.prompt_version,
        subgraph_size=len(last_scope.node_ids) if last_scope else 0,
        subgraph_tokens=last_scope.tokens if last_scope else 0,
        dropped=list(last_scope.dropped) if last_scope else [],
        widened=widened_ever, verify_predicates=verify_predicates,
        status=("ok" if got_real_response else "invalid"),
        stratum=query.stratum,
        plan_id=getattr(semantic_map, "plan_id", None), query_text=query.text)
