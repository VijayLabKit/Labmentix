"""
LLM Provider Abstraction.
Handles configuring and returning the active Language Model.
"""

from __future__ import annotations

from configs import settings

def get_llm(temperature: float | None = None):
    """
    Constructs and returns the active LLM instance based on settings.
    
    Raises:
        RuntimeError: If the required API key is not configured.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Set it in your environment "
            "(see .env.example) to use LLM-powered features."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    
    temp = temperature if temperature is not None else settings.GEMINI_TEMPERATURE

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=temp,
        google_api_key=settings.GEMINI_API_KEY,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
    )
