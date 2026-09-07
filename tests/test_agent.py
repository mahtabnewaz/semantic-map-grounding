"""Tests for src.agent. Mock provider only -- no real model is called.

Responses are scripted through the LLMClient transport seam (`_create`),
the documented test hook; the client stays provider='mock'.
"""

import json
import types

from src.agent import AgentConfig, resolve
from src.graph import Node, SemanticMap
from src.llm import LLMClient


# --- fixtures / helpers ----------------------------------------------------
def _query(text, stratum, terms):
    return types.SimpleNamespace(text=text, stratum=stratum, query_terms=terms)


def _json(node_id, target_phrase, *, label="", relation=None, anchor_id=None):
    return json.dumps({
        "target_phrase": target_phrase, "node_id": node_id,
        "node_label": label, "relation": relation, "anchor_id": anchor_id,
    })


def _scripted(responses):
    """A mock client whose _create yields the given texts in order, then
    repeats the last one. Records the call count on `.calls`."""
    client = LLMClient("mock", "m", base_backoff=0)
    seq = list(responses)
    client.calls = {"n": 0}

    def _create(system, user):
        i = min(client.calls["n"], len(seq) - 1)
        client.calls["n"] += 1
        return (seq[i], None, None)

    client._create = _create
    return client


def _office_kitchen_map():
    m = SemanticMap()
    m.plan_id = "P1"
    m.add_node(Node("r_off", "office", "room", (0.0, 0.0), attributes=["desk"]))
    m.add_node(Node("d0", "doorway", "doorway", (1.0, 0.0)))
    m.add_node(Node("r_kit", "kitchen", "room", (2.0, 0.0),
                    attributes=["sink", "fridge"]))
    m.add_edge("r_off", "d0")
    m.add_edge("d0", "r_kit")
    return m


def _chain_map():
    """r0-d0-r1-d1-r2-d2-r3, distinct labels that match no query term."""
    m = SemanticMap()
    m.plan_id = "C"
    labels = ["alpha", "bravo", "charlie", "delta"]
    for i, lab in enumerate(labels):
        m.add_node(Node(f"r{i}", lab, "room", (2.0 * i, 0.0)))
    for i in range(len(labels) - 1):
        d = f"d{i}"
        m.add_node(Node(d, "doorway", "doorway", (2.0 * i + 1.0, 0.0)))
        m.add_edge(f"r{i}", d)
        m.add_edge(d, f"r{i + 1}")
    return m


# --- tests -----------------------------------------------------------------
def test_valid_proposal_verifies_on_attempt_1():
    m = _office_kitchen_map()
    client = _scripted([_json("r_kit", "kitchen", label="kitchen")])
    cfg = AgentConfig(current_node="r_off", k_hop=2)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, cfg)
    assert res.node_id == "r_kit"
    assert res.abstained is False
    assert res.attempts == 1
    assert client.calls["n"] == 1
    assert res.attempt_records[0].verified is True
    assert res.prompt_version == "resolve_v1"
    assert res.encoding == "structured"


def test_semantic_snap_retries_and_abstains_at_n_max():
    m = _office_kitchen_map()
    # Model keeps returning the office for a kitchen query -> snap.
    client = _scripted([_json("r_off", "kitchen", label="office")])
    cfg = AgentConfig(current_node="r_off", n_max=3, k_hop=2)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, cfg)
    assert res.abstained is True
    assert res.node_id is None
    assert res.attempts == 3
    assert client.calls["n"] == 3
    assert all(r.verified is False for r in res.attempt_records)
    assert all(r.failed_predicate == "grounded" for r in res.attempt_records)


def test_verify_disabled_accepts_a_snap():
    m = _office_kitchen_map()
    client = _scripted([_json("r_off", "kitchen", label="office")])
    cfg = AgentConfig(current_node="r_off", n_max=3, verify_enabled=False)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, cfg)
    assert res.node_id == "r_off"       # accepted despite being wrong
    assert res.abstained is False
    assert res.attempts == 1
    assert client.calls["n"] == 1
    assert res.attempt_records[0].verified is None   # verification skipped


def test_n_max_1_makes_exactly_one_call():
    m = _office_kitchen_map()
    client = _scripted([_json("r_off", "kitchen")])   # would fail grounding
    cfg = AgentConfig(current_node="r_off", n_max=1)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, cfg)
    assert client.calls["n"] == 1
    assert res.attempts == 1
    assert res.abstained is True


def test_unparseable_output_counts_as_attempt():
    m = _office_kitchen_map()
    client = _scripted(["this is not json at all"])
    cfg = AgentConfig(current_node="r_off", n_max=2)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, cfg)
    assert res.attempts == 2
    assert client.calls["n"] == 2
    assert res.abstained is True
    assert all(r.parsed is False for r in res.attempt_records)
    assert all(r.failed_predicate == "parse" for r in res.attempt_records)


def test_expand_on_missing_widens_and_records():
    m = _chain_map()
    # Model returns an existing node that cannot ground the (unmatched)
    # query term -> grounding fails -> widen.
    client = _scripted([_json("r0", "zzznomatch", label="alpha")])
    cfg = AgentConfig(current_node="r0", n_max=3, k_hop=1,
                      expand_on_missing=True, token_budget=100_000)
    res = resolve(_query("find zzznomatch", "direct", ["zzznomatch"]),
                  m, client, cfg)
    assert res.widened is True
    assert res.attempt_records[0].widened is True
    sizes = [r.subgraph_size for r in res.attempt_records]
    assert sizes[1] > sizes[0]          # the hull grew after widening


def test_verification_uses_query_phrase_not_model_paraphrase():
    # The model picks the correct node but restates the target in words
    # outside the lexicon ("food preparation area"). Verification must ground
    # the QUERY's phrase ("cooking"), so the correct node still verifies.
    m = SemanticMap()
    m.plan_id = "P"
    m.add_node(Node("r_kit", "Kitchen", "room", (0.0, 0.0),
                    attributes=["cooking", "food", "meals"]))
    m.add_node(Node("d0", "doorway", "doorway", (1.0, 0.0)))
    m.add_node(Node("r_bath", "Bathroom", "room", (2.0, 0.0),
                    attributes=["washing", "shower"]))
    m.add_edge("r_kit", "d0")
    m.add_edge("d0", "r_bath")
    q = types.SimpleNamespace(text="which room is used for cooking?",
                              stratum="attribute", query_terms=["cooking"])
    client = _scripted([_json("r_kit", "food preparation area")])
    res = resolve(q, m, client, AgentConfig(current_node="r_bath", n_max=1))
    assert res.node_id == "r_kit"
    assert res.abstained is False
    # the model's paraphrase is preserved for diagnostics, not verified
    assert res.attempt_records[0].model_output["target_phrase"] \
        == "food preparation area"


def test_relational_uses_query_relation_and_anchor_not_model_claims():
    # The model names the correct neighbour but mislabels the relation and
    # anchor. Verification must use the QUERY's relation/anchor.
    m = _office_kitchen_map()
    q = types.SimpleNamespace(text="which room is next to the kitchen?",
                              stratum="relational", query_terms=["kitchen"],
                              relation="adjacent_to", anchor="r_kit")
    # model: right node r_off, but wrong relation/anchor claims.
    client = _scripted([_json("r_off", "somewhere", relation="nearest",
                              anchor_id="bogus")])
    res = resolve(q, m, client, AgentConfig(current_node="r_kit", n_max=1))
    assert res.node_id == "r_off"
    assert res.abstained is False
    rec = res.attempt_records[0]
    assert rec.verified is True
    assert rec.model_output["relation"] == "nearest"     # claim kept, not used


def test_functional_verification_is_structural_and_recorded():
    m = _office_kitchen_map()
    q = types.SimpleNamespace(text="where I'd brew coffee in the morning",
                              stratum="functional",
                              query_terms=["where I'd brew coffee in the morning"])
    client = _scripted([_json("r_kit", "kitchen")])
    res = resolve(q, m, client, AgentConfig(current_node="r_off", n_max=1))
    # verified by existence + reachability only; the functional phrasing would
    # not ground, but grounding is not applied to this stratum.
    assert res.node_id == "r_kit"
    assert res.abstained is False
    assert res.attempt_records[0].verified is True
    assert res.verify_predicates == "node_exists,reachable"


def test_verify_predicates_disabled_recorded():
    m = _office_kitchen_map()
    q = _query("go to the kitchen", "direct", ["kitchen"])
    client = _scripted([_json("r_kit", "kitchen")])
    res = resolve(q, m, client, AgentConfig(current_node="r_off", n_max=1,
                                            verify_enabled=False))
    assert res.verify_predicates == "disabled"


def test_invalid_status_on_transport_error():
    m = _office_kitchen_map()
    c = LLMClient("mock", "m", base_backoff=0)

    def boom(system, user):
        raise ConnectionError("rate limited / down")

    c._create = boom
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, c, AgentConfig(current_node="r_off", n_max=1))
    assert res.status == "invalid"       # infra failure, not a model answer
    assert res.abstained is True


def test_invalid_status_on_empty_content():
    m = _office_kitchen_map()
    c = LLMClient("mock", "m", base_backoff=0)
    c._create = lambda s, u: ("", None, None)   # empty reply (e.g. truncated)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, c, AgentConfig(current_node="r_off", n_max=1))
    assert res.status == "invalid"


def test_ok_status_on_real_response():
    m = _office_kitchen_map()
    client = _scripted([_json("r_kit", "kitchen")])
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, AgentConfig(current_node="r_off", n_max=1))
    assert res.status == "ok"            # real reply, even if it were wrong


def test_token_and_latency_accounting_sums_across_attempts():
    m = _office_kitchen_map()
    client = _scripted([_json("r_off", "kitchen")])   # snap, 3 attempts
    cfg = AgentConfig(current_node="r_off", n_max=3)
    res = resolve(_query("go to the kitchen", "direct", ["kitchen"]),
                  m, client, cfg)
    assert res.attempts == 3
    assert res.prompt_tokens == sum(r.prompt_tokens for r in res.attempt_records)
    assert res.completion_tokens == sum(
        r.completion_tokens for r in res.attempt_records)
    assert abs(res.latency_s
               - sum(r.latency_s for r in res.attempt_records)) < 1e-9
    assert res.prompt_tokens > 0 and res.completion_tokens > 0
