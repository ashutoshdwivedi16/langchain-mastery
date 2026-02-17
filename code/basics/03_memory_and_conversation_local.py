"""
Feature 3: Memory and Conversation
Demonstrates RunnableWithMessageHistory for stateful multi-turn conversations.
Covers buffer memory, sliding-window memory, and session isolation.
"""

from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

def check_ollama():
 import requests
 try:
 response = requests.get("http://localhost:11434/api/tags", timeout=2)
 return response.status_code == 200
 except:
 return False

# In-memory session store
session_store: dict = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
 """Return (or create) a ChatMessageHistory for the given session."""
 if session_id not in session_store:
 session_store[session_id] = ChatMessageHistory()
 return session_store[session_id]

# Feature 3A: Buffer Memory (full history)
def buffer_memory_example():
 print("\n" + "="*60)
 print("FEATURE 3A: Conversation Buffer Memory")
 print("="*60)
 print("Keeps the FULL conversation history.\n")

 llm = OllamaLLM(model="llama2", temperature=0.7)

 prompt = ChatPromptTemplate.from_messages([
 ("system", "You are a helpful assistant. Answer concisely in 1-2 sentences."),
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

 conversation = [
 "My name is Ashutosh.",
 "What is Python?",
 "What did I just tell you my name was?", # tests memory
 ]

 for turn in conversation:
 print(f"Human : {turn}")
 response = chain.invoke({"input": turn}, config=cfg)
 print(f"AI : {response}\n")

# Feature 3B: Window Memory (last k turns only)
class WindowChatHistory(ChatMessageHistory):
 """Keeps only the last k human+AI message pairs."""

 # Declare k as a Pydantic field so Pydantic v2 allows it
 k: int = 2

 def add_message(self, message: BaseMessage) -> None:
 super().add_message(message)
 # Keep only the last k*2 messages (k human + k AI)
 if len(self.messages) > self.k * 2:
 self.messages = self.messages[-(self.k * 2):]

window_store: dict = {}

def get_window_history(session_id: str) -> WindowChatHistory:
 if session_id not in window_store:
 window_store[session_id] = WindowChatHistory(k=2) # noqa: E501
 return window_store[session_id]

def window_memory_example():
 print("\n" + "="*60)
 print("FEATURE 3B: Window Memory (last 2 turns only)")
 print("="*60)
 print("Keeps only the last 2 exchanges — older context is dropped.\n")

 llm = OllamaLLM(model="llama2", temperature=0.7)

 prompt = ChatPromptTemplate.from_messages([
 ("system", "You are a helpful assistant. Answer concisely in 1-2 sentences."),
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

 conversation = [
 "My favourite color is blue.",
 "My favourite language is Python.",
 "My favourite food is pizza.",
 "What is my favourite color?", # should be forgotten (outside window)
 "What is my favourite language?", # should be remembered (within window)
 ]

 for turn in conversation:
 print(f"Human : {turn}")
 response = chain.invoke({"input": turn}, config=cfg)
 print(f"AI : {response}\n")

# Feature 3C: Multi-session isolation
def multi_session_example():
 print("\n" + "="*60)
 print("FEATURE 3C: Multiple Independent Sessions")
 print("="*60)
 print("Each session_id gets its own separate memory.\n")

 llm = OllamaLLM(model="llama2", temperature=0.7)

 prompt = ChatPromptTemplate.from_messages([
 ("system", "You are a helpful assistant. Answer concisely in 1 sentence."),
 MessagesPlaceholder(variable_name="history"),
 ("human", "{input}"),
 ])

 chain = RunnableWithMessageHistory(
 prompt | llm | StrOutputParser(),
 get_session_history,
 input_messages_key="input",
 history_messages_key="history",
 )

 # Session 1 — Alice
 print("--- Session: Alice ---")
 chain.invoke({"input": "My name is Alice."}, config={"configurable": {"session_id": "alice"}})
 r = chain.invoke({"input": "What is my name?"}, config={"configurable": {"session_id": "alice"}})
 print(f"Alice session → {r}\n")

 # Session 2 — Bob (separate memory, doesn't know about Alice)
 print("--- Session: Bob ---")
 chain.invoke({"input": "My name is Bob."}, config={"configurable": {"session_id": "bob"}})
 r = chain.invoke({"input": "What is my name?"}, config={"configurable": {"session_id": "bob"}})
 print(f"Bob session → {r}\n")

 # Cross-check: Alice's session doesn't know Bob
 print("--- Cross-check (Alice asking about Bob) ---")
 r = chain.invoke({"input": "Do you know anyone named Bob?"}, config={"configurable": {"session_id": "alice"}})
 print(f"Alice session → {r}")

def main():
 if not check_ollama():
 print("\n Ollama is not running!")
 print("Start it with: ollama serve")
 return

 print("\n Feature 3: Memory and Conversation (, )")

 buffer_memory_example()
 window_memory_example()
 multi_session_example()

 print("\n" + "="*60)
 print(" Memory and Conversation examples completed!")
 print("="*60)

if __name__ == "__main__":
 main()
