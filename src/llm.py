"""Unified Ollama/Groq client.

Ollama is primary (local, CPU-only on the target machine). Groq is used
only for the large-model comparison row. Both expose OpenAI-compatible
endpoints, so a single :class:`LLMClient` wraps both. A third "mock"
provider returns deterministic canned responses so the agent and its tests
run without any model. No prompt strings live here (CLAUDE.md keeps prompts
in one module); this is pure transport, accounting, and retry.

Reproducibility: temperature defaults to 0.0 with a fixed seed. Ollama
seeding is best-effort -- the server honours ``seed`` when it can, but CPU
kernels and model quantisation mean results may still vary slightly across
runs. Record the seed with every experiment regardless.

GROQ_API_KEY is read from .env (gitignored) via python-dotenv. A key is
never inlined; a missing key raises a clear error at construction, not a
401 mid-run.

Retry policy (CLAUDE.md cycle-depth ablation depends on this): transient
transport failures -- connection errors, 5xx, rate limits -- are retried
with exponential backoff, up to 3 attempts. Malformed *content* is never
retried here; validating and re-prompting is the agent's job, and counting
it as an LLM retry would corrupt the ablation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from .serialize import token_estimate

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq free tier is roughly 100K tokens/day on the large model. Ollama is
# local, so uncapped by default. None means "no cap".
DEFAULT_CAPS: dict[str, Optional[int]] = {
    "groq": 100_000, "ollama": None, "mock": None,
}

# Exception class names that indicate a transient transport failure worth
# retrying, whatever SDK version raised them.
_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "ServiceUnavailableError",
}


class LLMConfigError(RuntimeError):
    """Misconfiguration (unknown provider, missing key) -- fail at setup."""


class BudgetExceededError(RuntimeError):
    """A daily token cap has been reached; refuse rather than truncate."""


@dataclass
class LLMResponse:
    """One completion's result and accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    provider: str
    error: Optional[str] = None
    attempt_count: int = 1


def prompt_hash(system: str, user: str) -> str:
    """Stable hash of a (system, user) prompt pair; keys mock responses."""
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


def groq_api_keys() -> list[str]:
    """Groq keys from .env, in failover order: GROQ_API_KEY, GROQ_API_KEY_2,
    GROQ_API_KEY_3, ... The first is required. Keys are never inlined."""
    load_dotenv()
    keys = []
    primary = os.environ.get("GROQ_API_KEY")
    if primary:
        keys.append(primary)
    i = 2
    while os.environ.get(f"GROQ_API_KEY_{i}"):
        keys.append(os.environ[f"GROQ_API_KEY_{i}"])
        i += 1
    if not keys:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Add it to .env (gitignored) as "
            "GROQ_API_KEY=... (and optional GROQ_API_KEY_2 for failover) -- "
            "do not inline the key in source."
        )
    return keys


def _is_rate_limit(error: str) -> bool:
    e = (error or "").lower()
    return ("ratelimit" in e or "rate_limit" in e or "rate limit" in e
            or "429" in e or "quota" in e or "tokens per day" in e)


def _is_transient(exc: Exception) -> bool:
    """Whether ``exc`` is a retryable transport failure (not bad content)."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


# --------------------------------------------------------------------------
# token budget
# --------------------------------------------------------------------------
class TokenBudget:
    """Cumulative prompt/completion tokens per provider per day.

    Persisted to a JSON file so a long or resumed run stays within a daily
    cap. Warns once at ``warn_frac`` of the cap and refuses (raises
    :class:`BudgetExceededError`) once the cap is reached.
    """

    def __init__(self, path, caps: Optional[dict] = None,
                 warn_frac: float = 0.80):
        self.path = Path(path)
        self.caps = dict(DEFAULT_CAPS if caps is None else caps)
        self.warn_frac = warn_frac
        self._data: dict = self._load()
        self._warned: set[str] = set()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def _bucket(self, provider: str) -> dict:
        return (self._data.setdefault(provider, {})
                .setdefault(self._today(), {"prompt": 0, "completion": 0}))

    def total(self, provider: str) -> int:
        day = self._data.get(provider, {}).get(self._today())
        return 0 if day is None else day["prompt"] + day["completion"]

    def remaining(self, provider: str) -> Optional[int]:
        cap = self.caps.get(provider)
        return None if cap is None else max(0, cap - self.total(provider))

    def precheck(self, provider: str) -> None:
        """Refuse before a call if the cap is already spent."""
        cap = self.caps.get(provider)
        if cap is not None and self.total(provider) >= cap:
            raise BudgetExceededError(
                f"{provider} daily token cap reached: "
                f"{self.total(provider)}/{cap} on {self._today()}. "
                f"Refusing to continue rather than truncate."
            )

    def record(self, provider: str, prompt_tokens: int,
               completion_tokens: int) -> None:
        bucket = self._bucket(provider)
        bucket["prompt"] += int(prompt_tokens)
        bucket["completion"] += int(completion_tokens)
        self._save()
        cap = self.caps.get(provider)
        if cap:
            used = bucket["prompt"] + bucket["completion"]
            if (used >= cap * self.warn_frac and used < cap
                    and provider not in self._warned):
                self._warned.add(provider)
                warnings.warn(
                    f"{provider} token budget at {used}/{cap} "
                    f"({used / cap:.0%}) on {self._today()}",
                    stacklevel=2,
                )


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class LLMClient:
    """OpenAI-compatible client for Ollama, Groq, or an offline mock."""

    def __init__(self, provider: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 200, timeout: int = 120, seed: int = 0,
                 *, budget: Optional[TokenBudget] = None,
                 mock_responses: Optional[dict] = None,
                 base_backoff: float = 0.5, max_attempts: int = 3,
                 reasoning_effort: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed
        self.budget = budget
        self.mock_responses = dict(mock_responses or {})
        self.base_backoff = base_backoff
        self.max_attempts = max_attempts
        # For reasoning models (e.g. gpt-oss): with a small max_tokens the
        # reasoning trace can consume the whole budget and leave the answer
        # field empty. Set reasoning_effort ('low'/'medium'/'high') and a
        # generous max_tokens so the answer survives.
        self.reasoning_effort = reasoning_effort

        self._client = None                 # lazily created OpenAI() instance
        self._sleep: Callable[[float], None] = time.sleep

        load_dotenv()
        if self.provider == "ollama":
            self.base_url = os.environ.get("OLLAMA_BASE_URL",
                                           OLLAMA_DEFAULT_BASE_URL)
            self.api_key = "ollama"          # Ollama ignores the key
        elif self.provider == "groq":
            self.base_url = GROQ_BASE_URL
            self._api_keys = groq_api_keys()   # [GROQ_API_KEY, GROQ_API_KEY_2..]
            self._key_idx = 0
            self.api_key = self._api_keys[0]
        elif self.provider == "mock":
            self.base_url = None
            self.api_key = None
        else:
            raise LLMConfigError(
                f"unknown provider {provider!r}; "
                f"expected 'ollama', 'groq', or 'mock'"
            )

    # -- transport seam (overridable in tests) ----------------------------
    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout)
        return self._client

    def _create(self, system: str, user: str):
        """Perform one raw completion. Returns (text, prompt_tokens,
        completion_tokens); token counts may be None if the API omits usage.
        Raises transport exceptions (retried) but never inspects content."""
        if self.provider == "mock":
            return self._mock_create(system, user)
        client = self._ensure_client()
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            timeout=self.timeout,
        )
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        ptok = getattr(usage, "prompt_tokens", None) if usage else None
        ctok = getattr(usage, "completion_tokens", None) if usage else None
        return text, ptok, ctok

    def _mock_create(self, system: str, user: str):
        h = prompt_hash(system, user)
        text = self.mock_responses.get(h, f"MOCK_RESPONSE[{h[:12]}]")
        return text, None, None

    def _rotate_key(self) -> bool:
        """Switch to the next Groq key on rate-limit/quota. False if none."""
        keys = getattr(self, "_api_keys", [])
        if self._key_idx + 1 < len(keys):
            self._key_idx += 1
            self.api_key = keys[self._key_idx]
            self._client = None                   # force recreate with new key
            return True
        return False

    # -- public API -------------------------------------------------------
    def complete(self, system: str, user: str) -> LLMResponse:
        """Complete a prompt, with budget refusal, transient retry, and (for
        Groq) failover across keys on rate-limit/quota errors.

        Budget refusal raises :class:`BudgetExceededError` (fail loudly).
        Transport failures after ``max_attempts`` are returned as an
        :class:`LLMResponse` with ``error`` set, not raised, so the caller
        can record them. Malformed content is returned verbatim, unretried.
        """
        if self.budget is not None:
            self.budget.precheck(self.provider)   # raises if cap spent

        while True:
            result = self._complete_once(system, user)
            if (result.error is not None and self.provider == "groq"
                    and _is_rate_limit(result.error) and self._rotate_key()):
                continue                          # retry the whole call on next key
            return result

    def _complete_once(self, system: str, user: str) -> LLMResponse:
        start = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                text, ptok, ctok = self._create(system, user)
            except Exception as exc:                # noqa: BLE001
                if _is_transient(exc) and attempt < self.max_attempts:
                    self._sleep(self.base_backoff * (2 ** (attempt - 1)))
                    continue
                return LLMResponse(
                    text="", prompt_tokens=0, completion_tokens=0,
                    latency_s=time.monotonic() - start, model=self.model,
                    provider=self.provider,
                    error=f"{type(exc).__name__}: {exc}",
                    attempt_count=attempt,
                )
            # Success. Prefer real usage; fall back to the estimator.
            if ptok is None:
                ptok = token_estimate(f"{system}\n{user}")
            if ctok is None:
                ctok = token_estimate(text)
            if self.budget is not None:
                self.budget.record(self.provider, ptok, ctok)
            return LLMResponse(
                text=text, prompt_tokens=ptok, completion_tokens=ctok,
                latency_s=time.monotonic() - start, model=self.model,
                provider=self.provider, error=None, attempt_count=attempt,
            )
