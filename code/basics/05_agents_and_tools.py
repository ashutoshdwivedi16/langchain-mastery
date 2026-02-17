"""
Feature 5: Agents and Tools
Demonstrates how agents can use tools to solve complex problems.
Uses LangGraph's create_react_agent (LangChain 1.x / LangGraph 1.x).
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import datetime

load_dotenv()


# ── Custom tools ─────────────────────────────────────────────────────────────

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


# ── Helper ─────────────────────────────────────────────────────────────────

def _run_agent(graph, question: str, verbose: bool = True) -> str:
    """Run a LangGraph ReAct agent and return the final text answer."""
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    # Last message is the AI's final answer
    final = result["messages"][-1].content
    if verbose:
        print(f"\nFinal Answer: {final}")
    return final


# ── Examples ──────────────────────────────────────────────────────────────

def simple_agent_example():
    """Agent with basic tools"""
    print("\n" + "="*60)
    print("FEATURE 5A: Simple Agent with Tools")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [get_current_time, calculate, word_counter, text_reverser]
    graph = create_react_agent(llm, tools)

    questions = [
        "What time is it right now?",
        "What is 157 multiplied by 23?",
        "How many words are in this sentence: 'LangChain makes building AI applications easy and fun'?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)
        try:
            _run_agent(graph, question)
        except Exception as e:
            print(f"Error: {str(e)}")


def multi_step_agent_example():
    """Agent that needs to use multiple tools in sequence"""
    print("\n" + "="*60)
    print("FEATURE 5B: Multi-Step Agent Reasoning")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [get_current_time, calculate, word_counter, text_reverser]
    graph = create_react_agent(llm, tools)

    complex_question = (
        "First, calculate what is 15 + 27. Then, take that result and "
        "multiply it by 3. Finally, tell me how many digits are in the final result."
    )

    print(f"Question: {complex_question}\n")

    try:
        _run_agent(graph, complex_question)
    except Exception as e:
        print(f"Error: {str(e)}")


def custom_tool_with_class():
    """Creating tools using the Tool class"""
    print("\n" + "="*60)
    print("FEATURE 5C: Custom Tools with Tool Class")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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
            description="Useful for when you need to count the number of characters in a string.",
        ),
        Tool(
            name="VowelCounter",
            func=vowel_counter,
            description="Useful for counting how many vowels are in a text.",
        ),
        calculate,
    ]

    graph = create_react_agent(llm, tools)
    question = "How many vowels are in the word 'LangChain'?"

    print(f"Question: {question}\n")
    try:
        _run_agent(graph, question)
    except Exception as e:
        print(f"Error: {str(e)}")


def agent_with_memory_example():
    """Agent that maintains conversation memory across turns."""
    print("\n" + "="*60)
    print("FEATURE 5D: Agent with Memory")
    print("="*60)

    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [get_current_time, calculate, word_counter]

    # MemorySaver persists message history across invocations when the same
    # thread_id is used in the config.
    memory = MemorySaver()
    graph = create_react_agent(llm, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "agent-demo"}}

    conversation = [
        "Calculate 25 + 17 for me.",
        "Now multiply that result by 3.",
        "What was my first question?",
    ]

    print("Having a conversation with memory:\n")

    for user_input in conversation:
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print("="*60)
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            answer = result["messages"][-1].content
            print(f"\nAgent: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    simple_agent_example()
    multi_step_agent_example()
    custom_tool_with_class()
    agent_with_memory_example()


if __name__ == "__main__":
    main()
