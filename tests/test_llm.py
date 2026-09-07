"""Tests for src.llm. No real model is ever called (all via mock / seams)."""

import pytest

from src import llm
from src.llm import (
    BudgetExceededError,
    LLMClient,
    LLMConfigError,
    TokenBudget,
    prompt_hash,
)


# --- mock provider determinism --------------------------------------------
def test_mock_is_deterministic():
    c = LLMClient("mock", "m")
    r1 = c.complete("sys", "user")
    r2 = c.complete("sys", "user")
    assert r1.text == r2.text
    assert r1.provider == "mock"
    assert r1.error is None
    assert r1.attempt_count == 1
    # Different prompt -> different canned text.
    assert c.complete("sys", "other").text != r1.text


def test_mock_responses_are_keyed_by_prompt_hash():
    h = prompt_hash("S", "U")
    c = LLMClient("mock", "m", mock_responses={h: "canned answer"})
    assert c.complete("S", "U").text == "canned answer"
    # Unmapped prompt falls back to the deterministic stub.
    assert c.complete("S", "different").text.startswith("MOCK_RESPONSE[")


def test_mock_token_fallback_populates_counts():
    c = LLMClient("mock", "m")
    r = c.complete("system words here", "user words here")
    assert r.prompt_tokens > 0
    assert r.completion_tokens > 0


# --- retry behaviour -------------------------------------------------------
def test_retry_fires_on_connection_error():
    c = LLMClient("mock", "m", base_backoff=0)
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return ("recovered", None, None)

    c._create = flaky
    r = c.complete("s", "u")
    assert r.error is None
    assert r.text == "recovered"
    assert r.attempt_count == 3
    assert calls["n"] == 3


def test_no_retry_on_valid_response_with_bad_content():
    c = LLMClient("mock", "m", base_backoff=0)
    calls = {"n": 0}

    def bad_content(system, user):
        calls["n"] += 1
        return ("not-json garbage {", None, None)

    c._create = bad_content
    r = c.complete("s", "u")
    assert r.attempt_count == 1
    assert calls["n"] == 1
    assert r.text == "not-json garbage {"
    assert r.error is None


def test_no_retry_on_non_transient_exception():
    c = LLMClient("mock", "m", base_backoff=0)
    calls = {"n": 0}

    def boom(system, user):
        calls["n"] += 1
        raise ValueError("client bug, not transport")

    c._create = boom
    r = c.complete("s", "u")
    assert calls["n"] == 1
    assert r.attempt_count == 1
    assert r.error is not None and "ValueError" in r.error


def test_transient_exhausts_and_returns_error():
    c = LLMClient("mock", "m", base_backoff=0)
    calls = {"n": 0}

    def always(system, user):
        calls["n"] += 1
        raise ConnectionError("down")

    c._create = always
    r = c.complete("s", "u")
    assert calls["n"] == 3
    assert r.attempt_count == 3
    assert r.error is not None
    assert r.text == ""


# --- token budget ----------------------------------------------------------
def test_budget_refuses_past_cap(tmp_path):
    b = TokenBudget(tmp_path / "b.json", caps={"mock": 100})
    b.record("mock", 70, 0)           # 70 < 80% warn line
    b.precheck("mock")                # still ok
    b.record("mock", 40, 0)           # 110 >= 100 cap
    with pytest.raises(BudgetExceededError):
        b.precheck("mock")


def test_budget_refuses_through_client(tmp_path):
    b = TokenBudget(tmp_path / "b.json", caps={"mock": 10})
    b.record("mock", 10, 5)           # already over
    c = LLMClient("mock", "m", budget=b)
    with pytest.raises(BudgetExceededError):
        c.complete("s", "u")


def test_budget_warns_at_80_percent(tmp_path):
    b = TokenBudget(tmp_path / "b.json", caps={"mock": 100})
    with pytest.warns(UserWarning):
        b.record("mock", 85, 0)


def test_budget_persists_across_instances(tmp_path):
    path = tmp_path / "b.json"
    TokenBudget(path, caps={"mock": 100}).record("mock", 40, 10)
    reopened = TokenBudget(path, caps={"mock": 100})
    assert reopened.total("mock") == 50
    assert reopened.remaining("mock") == 50


def test_no_cap_never_refuses(tmp_path):
    b = TokenBudget(tmp_path / "b.json", caps={"ollama": None})
    b.record("ollama", 10_000_000, 0)
    b.precheck("ollama")              # must not raise
    assert b.remaining("ollama") is None


# --- provider configuration ------------------------------------------------
def test_ollama_config_needs_no_network():
    c = LLMClient("ollama", "llama3.1:8b")
    assert c.base_url.endswith("11434/v1")
    assert c.api_key == "ollama"
    assert c.provider == "ollama"


def test_missing_groq_key_raises_clear_error(monkeypatch):
    # Prevent .env from repopulating the key, then remove all Groq keys.
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for i in range(2, 6):
        monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)
    with pytest.raises(LLMConfigError) as exc:
        LLMClient("groq", "llama-3.1-70b-versatile")
    assert "GROQ_API_KEY" in str(exc.value)


def test_unknown_provider_raises():
    with pytest.raises(LLMConfigError):
        LLMClient("openai", "gpt-4")


def test_groq_rotates_key_on_rate_limit(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    c = LLMClient("groq", "m", base_backoff=0)
    seen = []

    def fake_create(system, user):
        seen.append(c.api_key)
        if c.api_key == "k1":
            raise RuntimeError("RateLimitError: rate_limit_exceeded (429)")
        return ("ok", 5, 5)

    c._create = fake_create
    r = c.complete("s", "u")
    assert r.error is None and r.text == "ok"
    assert c.api_key == "k2"          # failed over to the second key
    assert seen[0] == "k1" and "k2" in seen
