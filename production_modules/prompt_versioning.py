"""
Versioned prompt registry for the ticket classifier.

# PRODUCTION NOTE:
# In a real system, store prompts in a database (Postgres, DynamoDB) with full
# version history, author metadata, A/B test assignments, and rollback
# capability. Use a feature flag service to control active version per
# environment/customer segment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# In this demo, we keep prompts in a simple in-memory registry.
# Each entry is SYSTEM-instructions-only; the classifier module is responsible
# for injecting the ticket text and wiring structured-output / function-calling.
PROMPT_REGISTRY: dict[str, dict] = {
    "v1": {
        "version_id": "v1",
        "model": "openai/gpt-4o-mini",
        "created_at": "2024-01-01",
        "description": "Basic classification prompt",
        "template": "You are a customer support ticket classifier.",
    },
    "v2": {
        "version_id": "v2",
        "model": "openai/gpt-4o-mini",
        "created_at": "2024-03-01",
        "description": "Adds step-by-step reasoning instruction",
        "template": """You are an expert customer support ticket classifier.

Think step by step before classifying:
1. Identify the core problem the customer is facing.
2. Assess the emotional tone and urgency.
3. Determine which team is best equipped to resolve this.
4. Assign a priority based on business impact and customer sentiment.
5. Flag for human review if ambiguous or high-stakes.""",
    },
}


def get_prompt(version: str) -> dict:
    if version not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt version: {version}. Available: {list(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[version]


def get_latest() -> dict:
    latest_key = sorted(PROMPT_REGISTRY.keys())[-1]
    return PROMPT_REGISTRY[latest_key]


def get_active_version() -> str:
    """
    Return the currently active prompt version ID.

    Controlled by PROMPT_VERSION env var, defaulting to v2.
    """
    return os.getenv("PROMPT_VERSION", "v2")


def get_active_prompt() -> dict:
    """
    Return the full prompt record (metadata + template) for the active version.
    """
    return get_prompt(get_active_version())


def list_versions() -> list[dict]:
    """
    Return a list of lightweight metadata records for all known prompt versions.
    """
    return [
        {
            "version_id": v["version_id"],
            "description": v["description"],
            "model": v["model"],
            "created_at": v["created_at"],
        }
        for v in PROMPT_REGISTRY.values()
    ]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Available versions:")
    for v in list_versions():
        print(f"  {v['version_id']}: {v['description']} (model={v['model']})")

    print(f"\nActive version: {get_active_version()}")
    prompt = get_active_prompt()
    print(f"\nActive prompt template:\n{prompt['template']}")