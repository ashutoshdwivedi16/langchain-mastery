"""
Feature 5: Agents and Tools
Demonstrates how agents can use tools to solve complex problems
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
import datetime

load_dotenv()

# Define custom tools using the @tool decorator
@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time. Use this when you need to know what time it is."""
    return datetime.datetime.now().strftime(format)

@tool
def calculate(expression: str) -> str:
    """Performs mathematical calculations. Input should be a valid Python math expression like '5 + 3' or '10 * 2'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def word_counter(text: str) -> str:
    """Counts the number of words in the provided text."""
    word_count = len(text.split())
    return f"The text contains {word_count} words."

@tool
def text_reverser(text: str) -> str:
    """Reverses the given text."""
    return text[::-1]

def simple_agent_example():
    """Agent with basic tools"""
    print("\n" + "="*60)
    print("FEATURE 5A: Simple Agent with Tools")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create list of tools
    tools = [
        get_current_time,
        calculate,
        word_counter,
        text_reverser
    ]

    # Get the React prompt from hub
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    # Test the agent
    questions = [
        "What time is it right now?",
        "What is 157 multiplied by 23?",
        "How many words are in this sentence: 'LangChain makes building AI applications easy and fun'?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

        try:
            result = agent_executor.invoke({"input": question})
            print(f"\nFinal Answer: {result['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")

def multi_step_agent_example():
    """Agent that needs to use multiple tools"""
    print("\n" + "="*60)
    print("FEATURE 5B: Multi-Step Agent Reasoning")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [
        get_current_time,
        calculate,
        word_counter,
        text_reverser
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # Complex question requiring multiple tools
    complex_question = (
        "First, calculate what is 15 + 27. Then, take that result and "
        "multiply it by 3. Finally, tell me how many digits are in the final result."
    )

    print(f"Question: {complex_question}\n")

    try:
        result = agent_executor.invoke({"input": complex_question})
        print(f"\n{'='*60}")
        print(f"Final Answer: {result['output']}")
    except Exception as e:
        print(f"Error: {str(e)}")

def custom_tool_with_class():
    """Creating tools using the Tool class"""
    print("\n" + "="*60)
    print("FEATURE 5C: Custom Tools with Tool Class")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Define tools using Tool class
    def string_length(text: str) -> str:
        """Returns the length of the input string"""
        return f"The string has {len(text)} characters."

    def vowel_counter(text: str) -> str:
        """Counts the number of vowels in the text"""
        vowels = "aeiouAEIOU"
        count = sum(1 for char in text if char in vowels)
        return f"The text contains {count} vowels."

    tools = [
        Tool(
            name="StringLength",
            func=string_length,
            description="Useful for when you need to count the number of characters in a string."
        ),
        Tool(
            name="VowelCounter",
            func=vowel_counter,
            description="Useful for counting how many vowels are in a text."
        ),
        calculate
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    question = "How many vowels are in the word 'LangChain'?"

    print(f"Question: {question}\n")

    try:
        result = agent_executor.invoke({"input": question})
        print(f"\n{'='*60}")
        print(f"Final Answer: {result['output']}")
    except Exception as e:
        print(f"Error: {str(e)}")

def agent_with_memory_example():
    """Agent that maintains conversation memory"""
    print("\n" + "="*60)
    print("FEATURE 5D: Agent with Memory")
    print("="*60)

    from langchain.memory import ConversationBufferMemory
    from langchain.agents import create_structured_chat_agent

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [
        get_current_time,
        calculate,
        word_counter
    ]

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Get structured chat prompt
    prompt = hub.pull("hwchase17/structured-chat-agent")

    agent = create_structured_chat_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # Have a conversation
    conversation = [
        "Calculate 25 + 17 for me.",
        "Now multiply that result by 3.",
        "What was my first question?"
    ]

    print("Having a conversation with memory:\n")

    for user_input in conversation:
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print('='*60)

        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {result['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    simple_agent_example()
    multi_step_agent_example()
    custom_tool_with_class()
    agent_with_memory_example()

if __name__ == "__main__":
    main()
