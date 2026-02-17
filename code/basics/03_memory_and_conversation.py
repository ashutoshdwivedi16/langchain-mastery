"""
Feature 3: Memory and Conversation Management
Demonstrates how to maintain context in conversations using different memory types
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def buffer_memory_example():
    """Stores all conversation history in memory"""
    print("\n" + "="*60)
    print("FEATURE 3A: Conversation Buffer Memory")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create memory
    memory = ConversationBufferMemory(return_messages=True)

    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # Have a conversation
    print("\n--- Conversation Start ---")
    response1 = conversation.predict(input="Hi! My name is Alice.")
    print(f"AI: {response1}")

    response2 = conversation.predict(input="What's 5 + 5?")
    print(f"AI: {response2}")

    response3 = conversation.predict(input="What's my name?")
    print(f"AI: {response3}")

    # Show memory contents
    print("\n--- Memory Contents ---")
    print(memory.load_memory_variables({}))

def window_memory_example():
    """Keeps only the last K interactions in memory"""
    print("\n" + "="*60)
    print("FEATURE 3B: Conversation Buffer Window Memory")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Only keep last 2 interactions (4 messages: 2 human + 2 AI)
    memory = ConversationBufferWindowMemory(
        k=2,
        return_messages=True
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    print("\n--- Testing Window Memory (k=2) ---")
    conversation.predict(input="My favorite color is blue.")
    conversation.predict(input="I like to play tennis.")
    conversation.predict(input="I work as a software engineer.")

    # This should NOT remember the first message (favorite color)
    response = conversation.predict(input="What's my favorite color?")
    print(f"Q: What's my favorite color?")
    print(f"A: {response}")

    # This SHOULD remember recent messages
    response = conversation.predict(input="What do I do for work?")
    print(f"\nQ: What do I do for work?")
    print(f"A: {response}")

def summary_memory_example():
    """Summarizes conversation history to save tokens"""
    print("\n" + "="*60)
    print("FEATURE 3C: Conversation Summary Memory")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    print("\n--- Building Conversation History ---")
    conversation.predict(input="I'm planning a trip to Japan next month.")
    conversation.predict(input="I want to visit Tokyo, Kyoto, and Osaka.")
    conversation.predict(input="I'm interested in traditional temples and modern technology.")

    # Check the summary
    print("\n--- Memory Summary ---")
    memory_vars = memory.load_memory_variables({})
    print(memory_vars)

    # Ask a question based on history
    response = conversation.predict(input="Can you remind me where I want to go and what I'm interested in?")
    print(f"\n--- Response ---")
    print(response)

def main():
    buffer_memory_example()
    window_memory_example()
    summary_memory_example()

if __name__ == "__main__":
    main()
