import json

import openai  # noqa: F401
from openai import AsyncOpenAI

from api.config import get_settings


def _default_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Analyze a journal entry using an OpenAI-compatible LLM."""

    # Step 1: build client if not injected
    if client is None:
        client = _default_client()

    # Step 2 & 3: build messages and call the LLM
    response = await client.chat.completions.create(
        model=get_settings().openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a journal analysis assistant. "
                    "Analyze the journal entry and respond ONLY with a JSON object "
                    "containing exactly these keys:\n"
                    "  - sentiment: 'positive', 'negative', or 'neutral'\n"
                    "  - summary: exactly 2 sentences summarizing the entry\n"
                    "  - topics: a list of 2-4 key topics mentioned\n"
                    "No markdown, no code fences, just raw JSON."
                ),
            },
            {
                "role": "user",
                "content": entry_text,
            },
        ],
        response_format={"type": "json_object"},
    )

    # Step 4: parse the response
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Step 5: return with entry_id attached
    return {
        "entry_id": entry_id,
        "sentiment": parsed["sentiment"],
        "summary": parsed["summary"],
        "topics": parsed["topics"],
    }
