"""
Chat Service Module
Manages prompt assembly according to selected study modes and formats conversation history for Mistral API.
"""

from typing import List, Dict, Any, Generator, Tuple, Optional
from prompts.prompts import (
    GENERAL_STUDY_SYSTEM_PROMPT,
    EXPLAIN_CONCEPT_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    STUDY_NOTES_SYSTEM_PROMPT
)
from services.llm import generate_chat_response, stream_chat_response, DEFAULT_MISTRAL_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


MODE_SYSTEM_PROMPTS = {
    "General Chat": GENERAL_STUDY_SYSTEM_PROMPT,
    "Explain Concept": EXPLAIN_CONCEPT_SYSTEM_PROMPT,
    "Summarize": SUMMARIZE_SYSTEM_PROMPT,
    "Study Notes": STUDY_NOTES_SYSTEM_PROMPT,
}


def get_system_prompt_for_mode(mode: str) -> str:
    """Retrieve corresponding system prompt for the given mode name."""
    return MODE_SYSTEM_PROMPTS.get(mode, GENERAL_STUDY_SYSTEM_PROMPT)


def build_conversation_payload(
    messages: List[Dict[str, Any]],
    mode: str = "General Chat",
    max_history_turns: int = 10
) -> List[Dict[str, str]]:
    """
    Format stored conversation messages into Mistral-compatible message list with system prompt.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        mode: Selected study mode.
        max_history_turns: Number of recent turns to retain to avoid context overflow.

    Returns:
        List of dicts formatted for the Mistral Chat Completion API.
    """
    system_prompt = get_system_prompt_for_mode(mode)
    payload: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Slice recent messages (excluding any stray system messages in history)
    user_assistant_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant") and m["content"].strip()
    ]

    # Keep last N turns (1 turn = user + assistant = 2 messages)
    recent_msgs = user_assistant_msgs[-(max_history_turns * 2):]
    payload.extend(recent_msgs)

    return payload
