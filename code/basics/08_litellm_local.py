"""
Feature 8: LiteLLM — Universal LLM Interface
LiteLLM provides ONE unified API to call ANY LLM provider:
  - Ollama (local, free)
  - OpenAI
  - Anthropic Claude
  - Google Gemini
  - Groq (free tier)
  - Azure OpenAI
  - 100+ others

You switch providers by changing just the model name string.
No code changes needed.

100% local demo uses Ollama. Switch provider by changing model string.
"""

import os
import litellm
from litellm import completion
litellm.set_verbose = False   # suppress debug logs

OLLAMA_BASE = "http://localhost:11434"


def check_ollama():
    import requests
    try:
        return requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2).status_code == 200
    except:
        return False


# ── Feature 8A: Basic Completion ─────────────────────────────────────────────
def basic_completion():
    print("\n" + "="*60)
    print("FEATURE 8A: Basic Completion with LiteLLM")
    print("="*60)
    print("Same API — just change the model string to switch providers.\n")

    # With Ollama (local)
    response = completion(
        model="ollama_chat/llama2",
        messages=[{"role": "user", "content": "Say 'Hello from LiteLLM + Ollama!' and nothing else."}],
        api_base=OLLAMA_BASE,
    )

    print("Model      :", response.model)
    print("Response   :", response.choices[0].message.content)
    print("\n💡 To switch to OpenAI, just change model to: 'gpt-3.5-turbo'")
    print("   To switch to Claude, change to: 'claude-3-haiku-20240307'")
    print("   To switch to Groq,   change to: 'groq/llama3-8b-8192'")


# ── Feature 8B: Streaming ────────────────────────────────────────────────────
def streaming_completion():
    print("\n" + "="*60)
    print("FEATURE 8B: Streaming with LiteLLM")
    print("="*60)
    print("Streaming works the same way across ALL providers.\n")

    print("Question: What is Python in 2 sentences?\n")
    print("Response (streamed): ", end="", flush=True)

    response = completion(
        model="ollama_chat/llama2",
        messages=[{"role": "user", "content": "What is Python in exactly 2 sentences?"}],
        api_base=OLLAMA_BASE,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n")


# ── Feature 8C: Provider Switching Demo ──────────────────────────────────────
def provider_switching_demo():
    print("\n" + "="*60)
    print("FEATURE 8C: Provider Switching — Same Code, Different Model")
    print("="*60)
    print("This shows HOW you would switch providers (Ollama runs now).\n")

    # This is the SAME function for all providers
    def ask_llm(model: str, question: str, api_base: str = None) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": question}],
        }
        if api_base:
            kwargs["api_base"] = api_base
        response = completion(**kwargs)
        return response.choices[0].message.content

    question = "What is LangChain in one sentence?"

    # Currently running: Ollama
    print(f"Question: {question}\n")
    print("--- Running with: ollama_chat/llama2 (local) ---")
    answer = ask_llm("ollama_chat/llama2", question, api_base=OLLAMA_BASE)
    print(f"Answer: {answer}\n")

    print("--- Would run with: gpt-3.5-turbo (needs OPENAI_API_KEY) ---")
    print("    answer = ask_llm('gpt-3.5-turbo', question)")
    print("    # Returns same format — zero code change needed\n")

    print("--- Would run with: groq/llama3-8b-8192 (free Groq tier) ---")
    print("    answer = ask_llm('groq/llama3-8b-8192', question)")
    print("    # Set GROQ_API_KEY in .env — free at console.groq.com\n")

    print("--- Would run with: claude-3-haiku-20240307 (Anthropic) ---")
    print("    answer = ask_llm('claude-3-haiku-20240307', question)")
    print("    # Set ANTHROPIC_API_KEY in .env\n")


# ── Feature 8D: Usage Tracking ───────────────────────────────────────────────
def usage_tracking():
    print("\n" + "="*60)
    print("FEATURE 8D: Usage & Cost Tracking")
    print("="*60)
    print("LiteLLM tracks token usage across all providers.\n")

    response = completion(
        model="ollama_chat/llama2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "What is machine learning? One sentence."},
        ],
        api_base=OLLAMA_BASE,
    )

    print("Response  :", response.choices[0].message.content)
    print("\nUsage stats:")
    print(f"  Prompt tokens     : {response.usage.prompt_tokens}")
    print(f"  Completion tokens : {response.usage.completion_tokens}")
    print(f"  Total tokens      : {response.usage.total_tokens}")
    print("\n💡 For OpenAI, LiteLLM also calculates exact cost in USD.")


# ── Feature 8E: Multi-turn conversation ──────────────────────────────────────
def multi_turn_conversation():
    print("\n" + "="*60)
    print("FEATURE 8E: Multi-Turn Conversation with LiteLLM")
    print("="*60)
    print("Manual message history management — works with any provider.\n")

    messages = [
        {"role": "system", "content": "You are a concise assistant. Keep answers to 1-2 sentences."}
    ]

    conversation = [
        "My name is Ashutosh.",
        "What is Python?",
        "What is my name?",  # tests memory
    ]

    for user_input in conversation:
        messages.append({"role": "user", "content": user_input})
        print(f"Human : {user_input}")

        response = completion(
            model="ollama_chat/llama2",
            messages=messages,
            api_base=OLLAMA_BASE,
        )

        assistant_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})
        print(f"AI    : {assistant_reply}\n")


# ── Feature 8F: LiteLLM with LangChain ───────────────────────────────────────
def litellm_with_langchain():
    print("\n" + "="*60)
    print("FEATURE 8F: LiteLLM + LangChain Integration")
    print("="*60)
    print("Wrap LiteLLM in a RunnableLambda to use inside LCEL chains.\n")

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    # Wrap litellm.completion as a LangChain Runnable
    def litellm_invoke(prompt_text: str) -> str:
        response = completion(
            model="ollama_chat/llama2",
            messages=[{"role": "user", "content": prompt_text}],
            api_base=OLLAMA_BASE,
        )
        return response.choices[0].message.content

    litellm_llm = RunnableLambda(litellm_invoke)

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in exactly one sentence.",
    )

    # LCEL chain: prompt → LiteLLM → string output
    chain = prompt | litellm_llm | StrOutputParser()

    topics = ["recursion", "transformers in AI"]
    for topic in topics:
        result = chain.invoke({"topic": topic})
        print(f"Topic: {topic}")
        print(f"Answer: {result}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not check_ollama():
        print("❌ Ollama is not running! Start with: ollama serve")
        return

    print("\n💡 Feature 8: LiteLLM — Universal LLM Interface (local demo with Ollama)")
    print("   Same code works with OpenAI, Anthropic, Groq, Azure — just change model name!\n")

    basic_completion()
    streaming_completion()
    provider_switching_demo()
    usage_tracking()
    multi_turn_conversation()
    litellm_with_langchain()

    print("\n" + "="*60)
    print("✅ LiteLLM examples completed!")
    print("="*60)
    print("\n💡 Key takeaway:")
    print("   completion(model='ollama_chat/llama2', ...)  ← local, free")
    print("   completion(model='gpt-4o',             ...)  ← OpenAI")
    print("   completion(model='groq/llama3-8b-8192',...)  ← Groq free tier")
    print("   ZERO code changes — just the model string changes.")
    print("="*60)


if __name__ == "__main__":
    main()
