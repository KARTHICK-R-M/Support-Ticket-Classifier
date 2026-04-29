import json
import os
import logging

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from schema import TicketClassification

load_dotenv()

_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/free")

DEFAULT_SYSTEM_PROMPT = """
You are an expert customer support ticket classifier for an e-commerce company.

Classify the ticket into exactly one issue category, one assigned team, one priority,
one user sentiment, a confidence score, a short reasoning, and whether human review is needed.

Follow these strict business rules:

ISSUE CATEGORY RULES:
- refund_request: customer explicitly asks for refund, reversal, or money back.
- payment_issue: duplicate charge, failed payment, charged incorrectly, card/payment processing issue.
- delivery_issue: late delivery, shipment delay, package not received, courier issue.
- product_issue: damaged item, defective item, wrong item, missing part.
- account_issue: login, password, account access, profile/account settings.
- order_issue: order placement, cancellation, modification, tracking confusion not caused by delivery partner.
- other: use only if none of the above clearly fit.

TEAM ASSIGNMENT RULES:
- payment_issue -> payments_team
- refund_request -> customer_support
- delivery_issue -> logistics_team
- product_issue -> fulfillment_team
- account_issue -> tech_team
- order_issue -> customer_support
- other -> customer_support

SENTIMENT RULES:
- angry: explicit anger, frustration, threats, all caps, repeated urgency, or harsh tone.
- negative: unhappy or dissatisfied, but not strongly aggressive.
- neutral: factual, calm, or mixed with no strong emotion.
- positive: praise, thanks, or satisfaction.

PRIORITY RULES:
- critical: fraud, security risk, account takeover, or severe repeated financial harm.
- high: duplicate charges, urgent refunds, no delivery after long delay, very angry customer.
- medium: normal support issue affecting order/account/product.
- low: minor question, suggestion, or non-urgent request.

SPECIAL RULE:
- If a customer was charged twice and explicitly requests a refund, classify it as payment_issue and assign payments_team.
- Use refund_request only when the refund request is the main issue and there is no stronger payment-processing problem.

HUMAN REVIEW:
- Set requires_human_review = true if the ticket is ambiguous, contains multiple equally strong categories,
  contains possible fraud/legal escalation, or confidence is below 0.80.

Return only the structured classification.
""".strip()

# Used by fallback_retry on retry attempts — simpler, more conservative
SIMPLE_SYSTEM_PROMPT = (
    "You are a support ticket classifier for an e-commerce company. "
    "Choose exactly one issue category from: order_issue, payment_issue, delivery_issue, "
    "product_issue, account_issue, refund_request, other. "
    "Choose exactly one team from: fulfillment_team, payments_team, logistics_team, "
    "customer_support, tech_team. "
    "Choose exactly one sentiment from: positive, neutral, negative, angry. "
    "If duplicate charge or billing error is mentioned, prefer payment_issue and payments_team. "
    "If customer explicitly sounds aggressive or highly frustrated, use angry. "
    "Set requires_human_review=True if unsure."
)


# ---------------------------------------------------------------------------
# Approach 1: Function-calling
# ---------------------------------------------------------------------------


def classify_with_function_calling(
    ticket_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = _DEFAULT_MODEL,
) -> TicketClassification:
    llm = ChatOpenRouter(model=model, temperature=0)
    structured_llm = llm.with_structured_output(
        TicketClassification, method="function_calling"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Classify this support ticket:\n\n{ticket_text}"),
        ]
    )

    chain = prompt | structured_llm
    return chain.invoke({"ticket_text": ticket_text})


# ---------------------------------------------------------------------------
# Approach 2: JSON mode
# ---------------------------------------------------------------------------


def classify_with_json_mode(
    ticket_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = _DEFAULT_MODEL,
) -> TicketClassification:
    schema_json = json.dumps(TicketClassification.model_json_schema(), indent=2)
    # Escape braces so LangChain's template engine doesn't treat them as
    # variable placeholders (e.g. {"type": "string"} → {{"type": "string"}})
    schema_escaped = schema_json.replace("{", "{{").replace("}", "}}")
    full_system = f"{system_prompt}\n\nReturn ONLY valid JSON matching this schema:\n{schema_escaped}"

    llm = ChatOpenRouter(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", full_system),
            ("human", "Classify this support ticket:\n\n{ticket_text}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"ticket_text": ticket_text})
    raw = json.loads(response.content)
    logger = logging.getLogger(__name__)
    logger.debug("JSON mode raw output: %s", raw)
    return TicketClassification.model_validate(raw)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ticket = "I was charged twice for order #9981. Please refund immediately!"

    print("=== Approach 1: Function-calling ===")
    result1 = classify_with_function_calling(ticket)
    print(result1.model_dump_json(indent=2))

    '''
    print("\n=== Approach 2: JSON mode ===")
    result2 = classify_with_json_mode(ticket)
    print(result2.model_dump_json(indent=2))
    '''
