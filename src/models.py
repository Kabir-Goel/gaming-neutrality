"""Unified generation interface across the five providers in config.yaml.

One public function, `generate`, dispatches on provider name and normalizes
every response to the same dict shape so the rest of the pipeline never has
to care which SDK produced a completion.

Providers disagree about which decoding parameters they accept, and some
reject parameters they used to support. Rather than hardcode a capability
table that goes stale, `generate` learns the constraints from the 400s the
API returns: it drops the offending parameter, warns once on stderr, and
remembers the adjustment for the rest of the process. Anything dropped is a
change to the experimental condition, so read those warnings.
"""

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# provider -> (env var holding the key, base_url for OpenAI-compatible hosts)
_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "openai": ("OPENAI_API_KEY", None),
    "google": ("GOOGLE_API_KEY", None),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    "moonshot": ("MOONSHOT_API_KEY", "https://api.moonshot.ai/v1"),
}

# Decoding parameters that may be negotiated away by _adapt, in the order we
# check them against an error message.
_TUNABLE = ("temperature", "top_p", "max_tokens", "max_output_tokens")

# Learned per (provider, model): parameters the endpoint refused, and renames
# it asked for (e.g. max_tokens -> max_completion_tokens).
_DROPPED: dict[tuple[str, str], set[str]] = {}
_RENAMED: dict[tuple[str, str], dict[str, str]] = {}
_WARNED: set[tuple[str, str, str]] = set()


def _api_key(provider: str) -> str:
    """Return the key for `provider`, or raise naming the variable to set."""
    var, _ = _PROVIDERS[provider]
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(
            f"Missing API key for provider {provider!r}: set {var} in .env "
            f"(project root) or export it in the environment."
        )
    return key


# Substrings marking a 429 as an exhausted account rather than a rate limit:
# no amount of backoff clears these, so they are raised on the first attempt.
# Deliberately narrower than the word "quota" on its own — Google reports
# transient per-minute limits as "Quota exceeded for quota metric ...", and
# those genuinely are worth retrying.
_ACCOUNT_EXHAUSTED = (
    "insufficient balance",
    "insufficient_quota",
    "exceeded_current_quota",
    "credit balance",
    "plan and billing",
    "billing details",
    "suspended",
    "recharge",
)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_timeout(exc: BaseException) -> bool:
    """True for request timeouts from any SDK.

    All three SDKs layer over httpx, so the underlying error is either an
    httpx timeout or an SDK wrapper carrying one as its cause.
    """
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        return True
    if type(exc).__name__ in ("APITimeoutError", "DeadlineExceeded"):
        return True
    return isinstance(getattr(exc, "__cause__", None), httpx.TimeoutException)


def _is_account_exhausted(exc: BaseException) -> bool:
    """True when a 429 reports drained credit rather than request pacing."""
    message = str(exc).lower()
    return any(marker in message for marker in _ACCOUNT_EXHAUSTED)


def _is_retryable(exc: BaseException) -> bool:
    """True for timeouts, rate limits (429) and server errors (5xx).

    Anthropic and OpenAI expose `status_code`; google-genai exposes `code`.
    Both are ints on HTTP errors, which is what the isinstance guard checks
    (OpenAI also puts a string error slug on `.code`). A 429 that reports an
    exhausted balance or suspended account is not retried — backing off five
    times just delays a failure that will not resolve on its own.
    """
    if _is_timeout(exc):
        return True
    status = _status_of(exc)
    if status == 429:
        return not _is_account_exhausted(exc)
    return status is not None and 500 <= status < 600


_with_retries = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True,
)


@lru_cache(maxsize=None)
def _client(provider: str) -> Any:
    """Build (and memoize) the SDK client for a provider."""
    key = _api_key(provider)
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=key)
    if provider == "google":
        from google import genai

        return genai.Client(api_key=key)
    # openai, together and moonshot all speak the OpenAI chat-completions API.
    import openai

    _, base_url = _PROVIDERS[provider]
    return openai.OpenAI(api_key=key, base_url=base_url)


def _warn(key: tuple[str, str], param: str, message: str) -> None:
    if (*key, param) not in _WARNED:
        _WARNED.add((*key, param))
        print(f"[models] {key[1]}: {message}", file=sys.stderr)


def _apply_learned(key: tuple[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Re-apply adjustments this model has already forced us into."""
    params = {k: v for k, v in params.items() if k not in _DROPPED.get(key, ())}
    for old, new in _RENAMED.get(key, {}).items():
        if old in params:
            params[new] = params.pop(old)
    return params


def _adapt(key: tuple[str, str], params: dict[str, Any], exc: BaseException) -> bool:
    """Work around a 400 about an unsupported parameter.

    Returns True if `params` was changed and the call is worth retrying.
    Only 400s are considered: auth failures, unknown models and quota errors
    must surface to the caller untouched.
    """
    if _status_of(exc) != 400:
        return False
    message = str(exc)

    # A rename request, e.g. "Use 'max_completion_tokens' instead".
    for old in ("max_tokens", "max_output_tokens"):
        new = "max_completion_tokens"
        if old in params and new in message:
            _RENAMED.setdefault(key, {})[old] = new
            params[new] = params.pop(old)
            return True

    # Otherwise the endpoint named a parameter it will not accept at all.
    for param in _TUNABLE:
        named = f"'{param}'" in message or f"`{param}`" in message
        if named and param in params:
            _DROPPED.setdefault(key, set()).add(param)
            value = params.pop(param)
            _warn(
                key,
                param,
                f"rejected {param}={value}; sending the provider default "
                f"instead. Decoding is no longer matched across models.",
            )
            return True

    return False


def _call(
    key: tuple[str, str], params: dict[str, Any], fn: Callable[..., Any]
) -> tuple[Any, int]:
    """Invoke `fn(**params)`, negotiating away parameters it refuses.

    Returns the raw SDK response and the wall-clock latency of the successful
    attempt in milliseconds.
    """
    params = _apply_learned(key, params)
    for _ in range(len(_TUNABLE) + 1):
        t0 = time.perf_counter()
        try:
            response = fn(**params)
        except Exception as exc:
            if not _adapt(key, params, exc):
                raise
            continue
        return response, round((time.perf_counter() - t0) * 1000)
    raise RuntimeError(f"{key[1]}: could not find an accepted parameter set.")


@_with_retries
def _post_anthropic(model: str, prompt: str, **params: Any) -> Any:
    return _client("anthropic").messages.create(
        model=model, messages=[{"role": "user", "content": prompt}], **params
    )


@_with_retries
def _post_openai_compatible(
    provider: str, model: str, prompt: str, **params: Any
) -> Any:
    return _client(provider).chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **params
    )


@_with_retries
def _post_google(model: str, prompt: str, **params: Any) -> Any:
    from google.genai import types

    return _client("google").models.generate_content(
        model=model, contents=prompt, config=types.GenerateContentConfig(**params)
    )


def _enum_name(value: Any) -> str:
    """google-genai returns enums; the other SDKs return plain strings."""
    if value is None:
        return ""
    return getattr(value, "name", None) or str(value)


def generate(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Send a single-turn prompt to `model` and return a normalized record.

    Returns keys: text, model_version, finish_reason, latency_ms. Retries up
    to 5 attempts with exponential backoff on timeouts, rate limits and 5xx;
    anything else — auth failures, unknown models, and 429s reporting a
    drained balance — is raised immediately. Decoding parameters the endpoint
    rejects are dropped with a warning on stderr; see the module docstring.
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of "
            f"{', '.join(sorted(_PROVIDERS))}."
        )
    key = (provider, model)

    if provider == "anthropic":
        response, latency_ms = _call(
            key,
            {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
            lambda **p: _post_anthropic(model, prompt, **p),
        )
        return {
            "text": "".join(b.text for b in response.content if b.type == "text"),
            "model_version": response.model,
            "finish_reason": response.stop_reason or "",
            "latency_ms": latency_ms,
        }

    if provider == "google":
        response, latency_ms = _call(
            key,
            {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
            },
            lambda **p: _post_google(model, prompt, **p),
        )
        candidates = response.candidates or []
        parts = (candidates[0].content.parts or []) if candidates else []
        return {
            "text": "".join(p.text for p in parts if getattr(p, "text", None)),
            "model_version": response.model_version or model,
            "finish_reason": (
                _enum_name(candidates[0].finish_reason) if candidates else ""
            ),
            "latency_ms": latency_ms,
        }

    response, latency_ms = _call(
        key,
        {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
        lambda **p: _post_openai_compatible(provider, model, prompt, **p),
    )
    choice = response.choices[0]
    return {
        "text": choice.message.content or "",
        "model_version": response.model,
        "finish_reason": choice.finish_reason or "",
        "latency_ms": latency_ms,
    }


def effective_params(provider: str, model: str) -> dict[str, Any]:
    """What this process learned about a model's accepted decoding params.

    Useful for the methods section: it reports which requested parameters the
    endpoint refused and any renames it required.
    """
    key = (provider, model)
    return {
        "dropped": sorted(_DROPPED.get(key, ())),
        "renamed": dict(_RENAMED.get(key, {})),
    }


def _smoke_test() -> int:
    """Send 'Say hello' to every model in config.yaml; return an exit code."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    decoding = config["decoding"]

    failures = 0
    for entry in config["models"]:
        provider, name = entry["provider"], entry["name"]
        label = f"{entry['id']}  {provider:<10} {name}"
        try:
            result = generate(
                provider=provider,
                model=name,
                prompt="Say hello",
                temperature=decoding["temperature"],
                top_p=decoding["top_p"],
                max_tokens=decoding["max_tokens"],
            )
        except Exception as exc:  # noqa: BLE001 - the smoke test reports, not raises
            failures += 1
            print(f"FAIL  {label}\n        {type(exc).__name__}: {exc}\n")
            continue

        snippet = " ".join(result["text"].split())[:80]
        adjustments = effective_params(provider, name)
        print(
            f"OK    {label}\n"
            f"        version={result['model_version']}  "
            f"finish={result['finish_reason']}  "
            f"{result['latency_ms']}ms\n"
            f"        {snippet!r}"
        )
        if adjustments["dropped"] or adjustments["renamed"]:
            print(f"        adjusted: {adjustments}")
        print()

    total = len(config["models"])
    print(f"{total - failures}/{total} models reachable.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
