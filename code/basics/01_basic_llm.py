"""
Feature 1: Basic LLM Interaction
Demonstrates the simplest way to interact with an LLM using LangChain
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def basic_llm_example():
    """Basic LLM interaction - Hello World!"""
    print("\n" + "="*60)
    print("FEATURE 1: Basic LLM Interaction")
    print("="*60)

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    # Simple invoke
    response = llm.invoke("Say 'Hello World' in 5 different languages!")
    print(f"\nResponse:\n{response.content}")

    # Batch processing
    print("\n" + "-"*60)
    print("Batch Processing Example:")
    print("-"*60)

    messages = [
        "What is LangChain?",
        "What is a vector database?",
        "What is prompt engineering?"
    ]

    responses = llm.batch(messages)
    for i, response in enumerate(responses, 1):
        print(f"\nQ{i}: {messages[i-1]}")
        print(f"A{i}: {response.content[:100]}...")

if __name__ == "__main__":
    basic_llm_example()
