"""
Mistral AI LLM Service
Direct integration with the official Mistral AI Python SDK.
Handles authentication, chat completion, real-time streaming, and structured JSON generation.
"""

from typing import List, Dict, Any, Generator, Optional
import os
import json

from config import MISTRAL_API_KEY, DEFAULT_MISTRAL_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from utils.helpers import extract_json


class LLMError(Exception):
    """Base exception for LLM service failures."""
    pass


class LLMAuthError(LLMError):
    """Raised when API key is missing or invalid."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when Mistral API rate limit is exceeded."""
    pass


def get_mistral_client(api_key: Optional[str] = None):
    """
    Instantiate and return the official Mistral client.

    Args:
        api_key: Optional explicit API key. Falls back to config.MISTRAL_API_KEY.

    Raises:
        LLMAuthError: If no valid API key is found.
    """
    key = api_key or MISTRAL_API_KEY
    if not key or key == "your_mistral_api_key_here":
        raise LLMAuthError(
            "Mistral API key is missing. Please configure MISTRAL_API_KEY in your .env file "
            "or enter it in the sidebar settings."
        )

    try:
        from mistralai import Mistral
        return Mistral(api_key=key)
    except ImportError:
        raise ImportError(
            "mistralai package is not installed. Please run: pip install mistralai"
        )
    except Exception as e:
        raise LLMError(f"Failed to initialize Mistral client: {str(e)}")


def test_mistral_connection(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform a lightweight connectivity test with the Mistral API.

    Returns:
        Dict with 'success': bool, 'model': str, 'message': str.
    """
    try:
        client = get_mistral_client(api_key)
        # Test with a minimal 1-token query
        response = client.chat.complete(
            model=DEFAULT_MISTRAL_MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0.0
        )
        return {
            "success": True,
            "model": DEFAULT_MISTRAL_MODEL,
            "message": "Successfully connected to Mistral AI API."
        }
    except LLMAuthError as e:
        return {"success": False, "error": str(e), "type": "auth"}
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg:
            return {"success": False, "error": "Invalid Mistral API Key (401 Unauthorized).", "type": "auth"}
        elif "429" in err_msg:
            return {"success": False, "error": "Mistral API Rate Limit reached (429).", "type": "rate_limit"}
        return {"success": False, "error": f"Mistral API error: {err_msg}", "type": "api_error"}


def generate_chat_response(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MISTRAL_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None
) -> str:
    """
    Generate a non-streaming chat completion from Mistral AI.

    Args:
        messages: List of message dicts [{"role": "user"|"assistant"|"system", "content": "..."}]
        model: Mistral model identifier.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum response tokens.
        api_key: Optional API key override.

    Returns:
        Completed string response.
    """
    client = get_mistral_client(api_key)
    try:
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""
    except Exception as e:
        _handle_api_exception(e)


def stream_chat_response(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MISTRAL_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Stream tokens from Mistral AI in real time.

    Args:
        messages: List of message dicts [{"role": "user"|"assistant"|"system", "content": "..."}]
        model: Mistral model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.
        api_key: Optional API key override.

    Yields:
        Text chunks as they arrive from the API.
    """
    client = get_mistral_client(api_key)
    try:
        stream = client.chat.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        for event in stream:
            # Mistral SDK stream events deliver choices delta
            if hasattr(event, "data") and event.data.choices:
                chunk = event.data.choices[0].delta.content
                if chunk:
                    yield chunk
            elif hasattr(event, "choices") and event.choices:
                chunk = event.choices[0].delta.content
                if chunk:
                    yield chunk
    except Exception as e:
        _handle_api_exception(e)


def generate_structured_json(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MISTRAL_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prompt Mistral for structured JSON output and securely validate/parse it.

    Args:
        messages: List of message dicts.
        model: Model name.
        temperature: Low temperature for reliable structured syntax.
        max_tokens: Max output tokens.
        api_key: Optional API key override.

    Returns:
        Parsed Python dictionary.
    """
    client = get_mistral_client(api_key)
    try:
        # Request JSON object format
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content or ""
        parsed = extract_json(raw_content)
        if parsed is None:
            raise ValueError(f"Failed to parse model output into JSON: {raw_content[:200]}")
        return parsed
    except Exception as e:
        # If response_format json_object fails on older endpoints, fallback to standard completion + regex extractor
        try:
            raw = generate_chat_response(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        _handle_api_exception(e)


def _handle_api_exception(e: Exception):
    """Normalize Mistral and HTTP exceptions into user-friendly LLM errors."""
    err_str = str(e)
    if "401" in err_str or "Unauthorized" in err_str or "Invalid API Key" in err_str:
        raise LLMAuthError("Invalid Mistral API Key. Please verify your credentials.")
    elif "429" in err_str or "Rate limit" in err_str:
        raise LLMRateLimitError("Mistral AI Rate Limit reached. Please wait a moment before sending another request.")
    elif "model" in err_str.lower() and ("not found" in err_str.lower() or "does not exist" in err_str.lower()):
        raise LLMError(f"The selected model is unavailable: {err_str}")
    else:
        raise LLMError(f"Mistral AI API Error: {err_str}")
