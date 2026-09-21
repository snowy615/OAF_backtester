"""Shared Claude plumbing: client construction and a structured-output call."""

from __future__ import annotations

import os
from typing import Any, Type, TypeVar

from pydantic import BaseModel

DEFAULT_MODEL = "claude-opus-5"
# Server-side refusal fallback: if the primary model declines, the API re-runs the
# request on a fallback model inside the same call. Set OAF_CLAUDE_FALLBACKS=0 to disable.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def model_name() -> str:
    return os.environ.get("OAF_CLAUDE_MODEL", DEFAULT_MODEL)


def get_client():
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("the Claude features need the SDK: pip install 'oaf-backtester[llm]'") from e
    # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile.
    return anthropic.Anthropic()


def parse_structured(client, system: str, messages: list[dict[str, Any]], output: Type[T], max_tokens: int = 16000):
    """One structured-output call. Returns ``(parsed, response)``.

    The system prompt is long and identical across calls, so it is cached.
    """
    import anthropic

    kwargs: dict[str, Any] = dict(
        model=model_name(),
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        thinking={"type": "adaptive"},
        output_format=output,
    )
    if os.environ.get("OAF_CLAUDE_FALLBACKS", "1") != "0":
        kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
    try:
        response = client.beta.messages.parse(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMError("Claude authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Claude rate limit hit (the SDK already retried); try again shortly") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("could not reach the Claude API; check your connection") from e

    if response.stop_reason == "refusal":
        raise LLMError("Claude declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude's reply was cut off at max_tokens; simplify the request or raise the limit")
    if response.parsed_output is None:
        raise LLMError("Claude returned no structured output")
    return response.parsed_output, response
