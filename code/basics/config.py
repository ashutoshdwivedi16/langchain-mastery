"""
Shared config — reads LLM_PROVIDER from .env and returns the right LLM.

Usage in any example file:
    from config import get_llm, LLM_PROVIDER
    llm = get_llm()

.env settings:
    LLM_PROVIDER=ollama          # use Ollama (free, local) — DEFAULT
    LLM_PROVIDER=openai          # use OpenAI (needs OPENAI_API_KEY)

    OLLAMA_MODEL=llama2          # which Ollama model (default: llama2)
    OPENAI_MODEL=gpt-3.5-turbo   # which OpenAI model (default: gpt-3.5-turbo)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Read provider — default to ollama so it works out of the box
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llama2")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-3.5-turbo")
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.7"))


def get_llm(temperature: float = None):
    """
    Returns an LLM instance based on LLM_PROVIDER in .env.

    Examples:
        llm = get_llm()               # uses default temperature
        llm = get_llm(temperature=0)  # override temperature
    """
    t = temperature if temperature is not None else TEMPERATURE

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env"
            )
        print(f"🤖 Using OpenAI ({OPENAI_MODEL})")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=t)

    else:  # default: ollama
        from langchain_ollama import ChatOllama
        print(f"🦙 Using Ollama ({OLLAMA_MODEL}) — free & local")
        return ChatOllama(model=OLLAMA_MODEL, temperature=t)


def get_provider_info() -> str:
    if LLM_PROVIDER == "openai":
        return f"OpenAI ({OPENAI_MODEL})"
    return f"Ollama ({OLLAMA_MODEL})"
