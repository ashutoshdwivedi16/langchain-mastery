"""
Feature 1: Basic LLM Interaction
Demonstrates invoke, streaming, and multi-prompt patterns using a local Ollama model.
"""

from langchain_ollama import OllamaLLM
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def check_ollama():
 """Check if Ollama is accessible"""
 import requests
 try:
 response = requests.get("http://localhost:11434/api/tags", timeout=2)
 return response.status_code == 200
 except:
 return False

def basic_llm_example():
 """Basic LLM interaction with local Ollama"""
 print("\n" + "="*60)
 print("FEATURE 1: Basic Local LLM Interaction")
 print("="*60)

 if not check_ollama():
 print("\n Ollama is not running!")
 print("\nPlease start Ollama:")
 print(" 1. Run 'ollama serve' in a terminal")
 print(" 2. Or ensure the Ollama app is running")
 print(" 3. Make sure you have a model installed: ollama pull llama2")
 return

 print("\n Using Ollama (Local LLM - )")

 # Initialize LLM
 llm = OllamaLLM(
 model="llama2", # or "mistral", "phi", etc.
 temperature=0.7
 )

 # Simple invoke
 print("\n" + "-"*60)
 print("Simple Question:")
 print("-"*60)

 question = "Say 'Hello World' in 5 different languages!"
 print(f"\nQuestion: {question}\n")
 print("Response:")

 response = llm.invoke(question)
 print(response)

def streaming_example():
 """Demonstrates streaming responses"""
 print("\n" + "="*60)
 print("FEATURE 1B: Streaming Responses")
 print("="*60)

 if not check_ollama():
 print("\n Ollama is not running!")
 return

 # Initialize with streaming callback
 llm = OllamaLLM(
 model="llama2",
 temperature=0.7,
 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
 )

 print("\nQuestion: What is LangChain?\n")
 print("Streaming Response:")
 print("-"*60)

 llm.invoke("What is LangChain? Give a brief answer in 2-3 sentences.")

 print("\n" + "-"*60)

def multiple_questions():
 """Ask multiple questions"""
 print("\n" + "="*60)
 print("FEATURE 1C: Multiple Questions")
 print("="*60)

 if not check_ollama():
 print("\n Ollama is not running!")
 return

 llm = OllamaLLM(model="llama2", temperature=0.7)

 questions = [
 "What is machine learning in one sentence?",
 "What is a neural network in one sentence?",
 "What is Python in one sentence?"
 ]

 for i, question in enumerate(questions, 1):
 print(f"\n{i}. {question}")
 print("-"*60)
 response = llm.invoke(question)
 print(response)

def main():
 print("\n This example uses Ollama - a LLM!")
 print(" (after initial setup)\n")

 basic_llm_example()
 streaming_example()
 multiple_questions()

 print("\n" + "="*60)
 print(" Basic LLM examples completed!")
 print("="*60)
  print(" - Switch models: llm = OllamaLLM(model='mistral')")
 print(" - Adjust creativity: temperature=0.9 (higher = more creative)")
 print(" - Available models: Run 'ollama list' in terminal")
 print("="*60 + "\n")

if __name__ == "__main__":
 try:
 import requests
 main()
 except ImportError:
 print("Please install required packages:")
 print("pip install requests")
