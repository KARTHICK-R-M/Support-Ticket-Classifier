import logging
import os

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from schema import TicketClassification, IssueCategory, TeamOwner, Priority, Sentiment
from production_modules.validate_response import validate_classification
from production_modules.structured_output import (
    classify_with_function_calling,
    SIMPLE_SYSTEM_PROMPT,
)
from production_modules.prompt_versioning import get_active_prompt


load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/free")


SAFE_CLASSIFICATION = TicketClassification(
    issue_category=IssueCategory.OTHER,
    assigned_team=TeamOwner.CUSTOMER_SUPPORT,
    priority=Priority.MEDIUM,
    user_sentiment=Sentiment.NEUTRAL,
    confidence_score=0.0,
    reasoning="Automatic fallback: classification failed after all retries",
    requires_human_review=True,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def classify_with_retry(ticket_text: str, model: str = _DEFAULT_MODEL) -> TicketClassification:
    attempt = classify_with_retry.statistics.get("attempt_number", 1)

    if attempt > 1:
        logger.warning("Retry attempt %d — switching to simple system prompt", attempt)
        system_prompt = SIMPLE_SYSTEM_PROMPT
    else:
        system_prompt = get_active_prompt()["template"]

    result = classify_with_function_calling(
        ticket_text=ticket_text,
        system_prompt=system_prompt,
        model=model,
    )

    validation = validate_classification(result)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.error_details))

    return validation.validated_classification


def classify_with_fallback(ticket_text: str, model: str = _DEFAULT_MODEL) -> TicketClassification:
    try:
        return classify_with_retry(ticket_text, model)
    except Exception as exc:
        logger.error("All retries exhausted: %s. Returning safe classification.", exc)
        return SAFE_CLASSIFICATION