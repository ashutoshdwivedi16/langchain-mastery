"""
Feature 5: Agents and Tools - LOCAL VERSION (No API Required)
Uses Ollama (llama3.2) for local LLM inference.
Demonstrates LangGraph ReAct agents with custom tools.

NOTE: Uses llama3.2 instead of llama2 — better at tool/function calling.
"""

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent as create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import datetime


def check_ollama():
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# ── Custom tools ─────────────────────────────────────────────────────────

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time. Use this when you need to know what time or date it is."""
    return datetime.datetime.now().strftime(format)


@tool
def calculate(expression: str) -> str:
    """
    Performs mathematical calculations.
    Input must be a valid Python math expression like '5 + 3', '10 * 2', '100 / 4'.
    Only supports basic arithmetic: +, -, *, /, **, //, %.
    """
    try:
        # Safe eval — only allow basic math
        allowed = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow,
        }
        result = eval(expression, allowed, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """Counts the number of words in the provided text."""
    count = len(text.split())
    return f"The text contains {count} words."


@tool
def string_reverser(text: str) -> str:
    """Reverses the characters in the given text string."""
    return f"Reversed: {text[::-1]}"


@tool
def vowel_counter(text: str) -> str:
    """Counts how many vowels (a,e,i,o,u) are in the given text."""
    count = sum(1 for c in text.lower() if c in "aeiou")
    return f"The text contains {count} vowels."


# ── Helper ────────────────────────────────────────────────────────────────

def run_agent(graph, question: str) -> str:
    """Invoke a LangGraph agent and print the final answer."""
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    print(f"\nAgent: {answer}")
    return answer


# ── Examples ──────────────────────────────────────────────────────────────

def simple_tool_calling():
    """Agent uses a single tool to answer"""
    print("\n" + "="*60)
    print("FEATURE 5A: Simple Tool Calling")
    print("="*60)

    # llama3.2 is better at tool use than llama2
    llm = ChatOllama(model="llama3.2", temperature=0)
    tools = [get_current_time, calculate, word_counter]
    graph = create_react_agent(llm, tools)

    questions = [
        "What is the current date and time?",
        "What is 144 divided by 12?",
        "How many words are in this sentence: 'LangChain is a powerful framework for AI applications'?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Human: {q}")
        run_agent(graph, q)


def multi_tool_reasoning():
    """Agent chains multiple tool calls to solve a problem"""
    print("\n" + "="*60)
    print("FEATURE 5B: Multi-Step Tool Reasoning")
    print("="*60)

    llm = ChatOllama(model="llama3.2", temperature=0)
    tools = [calculate, word_counter, string_reverser, vowel_counter]
    graph = create_react_agent(llm, tools)

    question = "Calculate 25 multiplied by 4, then tell me how many vowels are in the word 'LangChain'."
    print(f"\nHuman: {question}")
    run_agent(graph, question)


def agent_with_memory():
    """Agent maintains context across multiple turns using MemorySaver"""
    print("\n" + "="*60)
    print("FEATURE 5C: Agent with Persistent Memory")
    print("="*60)
    print("Same thread_id = agent remembers previous messages.\n")

    llm = ChatOllama(model="llama3.2", temperature=0)
    tools = [calculate, word_counter, get_current_time]

    memory = MemorySaver()
    graph = create_react_agent(llm, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "local-agent-demo"}}

    conversation = [
        "Calculate 50 plus 30.",
        "Now multiply that result by 2.",
        "What was the very first calculation I asked you to do?",
    ]

    for user_msg in conversation:
        print(f"Human : {user_msg}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )
        answer = result["messages"][-1].content
        print(f"Agent : {answer}\n")


def main():
    if not check_ollama():
        print("\n❌ Ollama is not running!")
        print("Start it with: ollama serve")
        return

    print("\n💡 Feature 5: Agents and Tools (100% Local, No API needed!)")
    print("   Using llama3.2 — better at tool calling than llama2\n")

    simple_tool_calling()
    multi_tool_reasoning()
    agent_with_memory()

    print("\n" + "="*60)
    print("✅ Agents and Tools examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
