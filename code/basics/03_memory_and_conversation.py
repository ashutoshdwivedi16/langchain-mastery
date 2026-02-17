"""
Feature 3: Memory and Conversation Management
Modern LangChain 1.x approach using RunnableWithMessageHistory
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Shared session store ──────────────────────────────────────────────────────
store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def buffer_memory_example():
    """Full history — every message retained."""
    print("\n" + "="*60)
    print("FEATURE 3A: Full Conversation History")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "buffer-demo"}}

    print("\n--- Conversation ---")
    turns = [
        "Hi! My name is Alice.",
        "What is 5 + 5?",
        "What is my name?",   # tests memory retention
    ]
    for turn in turns:
        response = chain.invoke({"input": turn}, config=cfg)
        print(f"Human : {turn}")
        print(f"AI     : {response}\n")

    print("--- Message history ---")
    for msg in get_session_history("buffer-demo").messages:
        print(f"  [{msg.type}] {msg.content[:80]}")


def window_memory_example():
    """Sliding-window — keeps only the last k=2 exchanges."""
    print("\n" + "="*60)
    print("FEATURE 3B: Sliding-Window Memory  (k=2 turns)")
    print("="*60)

    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage

    class WindowChatHistory(BaseChatMessageHistory):
        """Retains only the most recent `k` human+AI pairs."""
        def __init__(self, k: int = 2):
            self.k = k
            self._messages: list[BaseMessage] = []

        @property
        def messages(self) -> list[BaseMessage]:
            return self._messages[-(self.k * 2):]

        def add_messages(self, messages: list[BaseMessage]) -> None:
            self._messages.extend(messages)

        def clear(self) -> None:
            self._messages = []

    window_store: dict[str, WindowChatHistory] = {}

    def get_window_history(session_id: str) -> WindowChatHistory:
        if session_id not in window_store:
            window_store[session_id] = WindowChatHistory(k=2)
        return window_store[session_id]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_window_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "window-demo"}}

    facts = [
        "My favourite colour is blue.",
        "I play tennis on weekends.",
        "I work as a software engineer.",
    ]
    for fact in facts:
        chain.invoke({"input": fact}, config=cfg)
        print(f"  Stored: {fact}")

    print()
    for question in ["What is my favourite colour?", "What do I do for work?"]:
        answer = chain.invoke({"input": question}, config=cfg)
        print(f"Q: {question}")
        print(f"A: {answer}\n")


def summary_memory_example():
    """Summary placeholder — demonstrates manual summarisation pattern."""
    print("\n" + "="*60)
    print("FEATURE 3C: Manual Summary Pattern")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Build up a history manually, then summarise it
    history = ChatMessageHistory()

    facts = [
        "I'm planning a trip to Japan next month.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "I'm interested in traditional temples and modern technology.",
    ]

    print("Adding conversation history...")
    for fact in facts:
        history.add_user_message(fact)
        history.add_ai_message("Got it, noted!")
        print(f"  + {fact}")

    # Summarise with a one-shot call
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise the following conversation in 2-3 sentences."),
        MessagesPlaceholder(variable_name="history"),
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"history": history.messages})

    print(f"\nSummary:\n{summary}")


def main():
    buffer_memory_example()
    window_memory_example()
    summary_memory_example()


if __name__ == "__main__":
    main()
