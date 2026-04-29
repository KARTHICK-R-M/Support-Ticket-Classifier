"""
Token cost estimation and session-level cost tracking.

# PRODUCTION NOTE:
# In a real system, prefer provider-reported usage/cost over local estimation.
# For OpenRouter, usage accounting can be returned directly in responses.
# Persist cost data per user/org/request, set up budget alerts, expose a cost
# dashboard, and integrate with billing systems for reconciliation.
"""

from dataclasses import dataclass
import os
from typing import Any

import tiktoken


# Prices in USD per 1,000 tokens.
# Example: GPT-4o mini is commonly listed at $0.15 / 1M input and
# $0.60 / 1M output, which equals $0.00015 / 1K and $0.00060 / 1K. [web:298][web:300]
PRICING_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "openai/gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4-turbo": {"input": 0.0100, "output": 0.0300},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/free")


@dataclass
class CostInfo:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class SessionCostTracker:
    """Accumulates estimated cost across multiple LLM calls in a process."""

    def __init__(self) -> None:
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0

    def record(self, cost_info: CostInfo) -> None:
        self._total_input_tokens += cost_info.input_tokens
        self._total_output_tokens += cost_info.output_tokens
        self._total_cost_usd += cost_info.total_cost_usd
        self._call_count += 1

    @property
    def summary(self) -> dict[str, int | float]:
        return {
            "calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
        }


# Module-level session tracker (reset per process)
session_tracker = SessionCostTracker()


def _tokenizer_model_name(model: str) -> str:
    """
    Map routed/provider-style model IDs to a tokenizer-compatible OpenAI name
    where possible.
    """
    if model in {"openrouter/free", "openai/gpt-4o-mini"}:
        return "gpt-4o-mini"
    if model == "openai/gpt-4o":
        return "gpt-4o"
    return model


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count tokens in a string using tiktoken.

    For provider-routed names that tiktoken doesn't recognize (e.g. openrouter/free),
    fall back to a compatible tokenizer or cl100k_base.
    """
    normalized_model = _tokenizer_model_name(model)
    try:
        encoding = tiktoken.encoding_for_model(normalized_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    record_to_session: bool = True,
) -> CostInfo:
    """
    Estimate cost for an LLM call and optionally record it to the session tracker.

    If the exact model is unknown in the pricing table, fall back conservatively
    to GPT-4o mini pricing for demo purposes.
    """
    pricing = PRICING_PER_1K.get(model)

    if pricing is None:
        normalized_model = _tokenizer_model_name(model)
        pricing = PRICING_PER_1K.get(normalized_model, PRICING_PER_1K["gpt-4o-mini"])

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total = input_cost + output_cost

    info = CostInfo(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(total, 6),
    )

    if record_to_session:
        session_tracker.record(info)

    return info


def cost_info_from_provider_usage(
    model: str,
    usage: dict[str, Any],
    record_to_session: bool = True,
) -> CostInfo:
    """
    Build CostInfo from provider-reported usage when available.

    Expected shape is flexible, but typically:
    {
        "prompt_tokens": ...,
        "completion_tokens": ...,
        "cost": ...
    }

    If direct cost is unavailable, falls back to local estimation.
    """
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))

    if "cost" in usage and usage["cost"] is not None:
        total_cost = float(usage["cost"])
        info = CostInfo(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=round(total_cost, 6),
        )
        if record_to_session:
            session_tracker.record(info)
        return info

    return calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        record_to_session=record_to_session,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Classify this ticket: My order hasn't arrived after 2 weeks!"
    response = '{"issue_category": "delivery_issue", "priority": "high"}'

    in_tokens = count_tokens(prompt, model="openai/gpt-4o-mini")
    out_tokens = count_tokens(response, model="openai/gpt-4o-mini")
    cost = calculate_cost("openai/gpt-4o-mini", in_tokens, out_tokens)

    print(f"Input tokens : {cost.input_tokens}")
    print(f"Output tokens: {cost.output_tokens}")
    print(f"Input cost   : ${cost.input_cost_usd:.6f}")
    print(f"Output cost  : ${cost.output_cost_usd:.6f}")
    print(f"Total cost   : ${cost.total_cost_usd:.6f}")
    print(f"Session total: {session_tracker.summary}")
