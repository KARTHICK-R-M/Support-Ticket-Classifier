"""
LangGraph pipeline for ticket classification.

Each node delegates to a production module — no LLM wiring lives here.

Node order: pii_redact → injection_check → classify → validate → cost_log
                                      ↓ (fail)
                                   fallback → cost_log
"""

import logging
import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from schema import TicketState
from production_modules.pii_redaction import redact_pii
from production_modules.prompt_injection import check_injection
from production_modules.prompt_versioning import get_active_prompt, get_active_version
from production_modules.structured_output import classify_with_function_calling
from production_modules.validate_response import validate_classification
from production_modules.cost_calculator import calculate_cost, count_tokens
from production_modules.fallback_retry import classify_with_fallback, SAFE_CLASSIFICATION

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/free")

# ---------------------------------------------------------------------------
# Node: pii_redact
# ---------------------------------------------------------------------------
def pii_redact_node(state: TicketState) -> dict:
    result = redact_pii(state.raw_ticket)
    return {
        "redacted_ticket": result.redacted_text,
        "pii_detected": result.pii_detected,
        "detected_entity_types": result.detected_entity_types,
    }

# ---------------------------------------------------------------------------
# Node: injection_check
# ---------------------------------------------------------------------------
def injection_check_node(state: TicketState) -> dict:
    check = check_injection(state.redacted_ticket or state.raw_ticket)
    if not check.is_safe:
        logger.warning("Injection detected: %s", check.detected_pattern)
        # Short‑circuit the rest of the pipeline with a safe classification
        return {
            "injection_blocked": True,
            "error": f"Injection detected: {check.detected_pattern}",
            "classification": SAFE_CLASSIFICATION,
            "validation_status": "blocked",
        }
    # No changes if safe; state flows through unchanged
    return {}

# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------
def classify_node(state: TicketState) -> dict:
    if state.injection_blocked:
        # Injection already blocked – skip classification
        return {}

    active = get_active_prompt()
    # This ensures the active version is loaded and logged; return value unused here
    get_active_version()

    ticket_text = state.redacted_ticket or state.raw_ticket

    try:
        classification = classify_with_function_calling(
            ticket_text=ticket_text,
            system_prompt=active["template"],
            model=DEFAULT_MODEL,
        )
        return {
            "classification": classification,
            "prompt_version": active.get("id") or active.get("version_id"),
            "error": None,
        }
    except Exception as exc:
        logger.error("classify_node failed: %s", exc)
        # Leave classification unset; validate node / fallback will handle it
        return {
            "error": f"classify_node failed: {exc}",
        }

# ---------------------------------------------------------------------------
# Node: validate
# ---------------------------------------------------------------------------
def validate_node(state: TicketState) -> dict:
    if state.injection_blocked:
        # Already blocked; nothing to validate
        return {}

    raw = state.classification

    result = validate_classification(raw)
    return {
        "validation_status": "pass" if result.is_valid else "fail",
        "error": None if result.is_valid else "; ".join(result.error_details),
        "classification": result.validated_classification if result.is_valid else raw,
    }

# ---------------------------------------------------------------------------
# Node: fallback
# ---------------------------------------------------------------------------
def fallback_node(state: TicketState) -> dict:
    logger.warning("Entering fallback node — delegating to classify_with_fallback")
    ticket_text = state.redacted_ticket or state.raw_ticket

    classification = classify_with_fallback(ticket_text=ticket_text, model=DEFAULT_MODEL)
    validation_status = "fallback_pass" if classification.confidence_score > 0.0 else "fallback_safe"

    return {
        "classification": classification,
        "validation_status": validation_status,
        "error": None,
    }

# ---------------------------------------------------------------------------
# Node: cost_log
# ---------------------------------------------------------------------------
def cost_log_node(state: TicketState) -> dict:
    ticket_text = state.redacted_ticket or state.raw_ticket
    classification = state.classification

    input_tokens = count_tokens(ticket_text, DEFAULT_MODEL)
    output_tokens = count_tokens(
        classification.model_dump_json() if classification else "",
        DEFAULT_MODEL,
    )
    cost_info = calculate_cost(DEFAULT_MODEL, input_tokens, output_tokens)

    if os.getenv("LOG_COSTS", "true").lower() == "true":
        logger.info(
            "Cost — model: %s | in: %d | out: %d | total: $%.6f",
            cost_info.model,
            cost_info.input_tokens,
            cost_info.output_tokens,
            cost_info.total_cost_usd,
        )

    return {
        "cost_info": {
            "model": cost_info.model,
            "input_tokens": cost_info.input_tokens,
            "output_tokens": cost_info.output_tokens,
            "total_cost_usd": cost_info.total_cost_usd,
        },
    }

# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------
def route_after_validate(state: TicketState) -> str:
    if state.injection_blocked:
        return "cost_log"
    if state.validation_status == "pass":
        return "cost_log"
    return "fallback"

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(TicketState)

    builder.add_node("pii_redact", pii_redact_node)
    builder.add_node("injection_check", injection_check_node)
    builder.add_node("classify", classify_node)
    builder.add_node("validate", validate_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("cost_log", cost_log_node)

    builder.add_edge(START, "pii_redact")
    builder.add_edge("pii_redact", "injection_check")
    builder.add_edge("injection_check", "classify")
    builder.add_edge("classify", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "cost_log": "cost_log",
            "fallback": "fallback",
        },
    )
    builder.add_edge("fallback", "cost_log")
    builder.add_edge("cost_log", END)

    return builder.compile()

graph = build_graph()

def run_pipeline(ticket_text: str, channel: str = "web_form") -> TicketState:
    # Construct initial TicketState object instead of a dict
    initial_state = TicketState(
        raw_ticket=ticket_text,
        channel=channel,
        redacted_ticket=None,
        classification=None,
        validation_status=None,
        cost_info=None,
        error=None,
        pii_detected=False,
        detected_entity_types=[],
        prompt_version=None,
        injection_blocked=False,
    )
    return graph.invoke(initial_state)