"""
Feature 10: Multi-Agent Patterns
Demonstrates parallel agents, sequential pipelines, supervisor routing,
and shared memory across agents using LangGraph.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent as create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import datetime

def check_ollama():
 import requests
 try:
 return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
 except:
 return False

# Shared Tools

@tool
def calculate(expression: str) -> str:
 """Performs mathematical calculations. Input: a Python math expression like '5 + 3'."""
 try:
 result = eval(expression, {"__builtins__": {}}, {})
 return f"Result: {result}"
 except Exception as e:
 return f"Error: {e}"

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
 """Returns the current date and time."""
 return datetime.datetime.now().strftime(format)

@tool
def word_counter(text: str) -> str:
 """Counts words in the given text."""
 return f"{len(text.split())} words"

@tool
def text_summarizer(text: str) -> str:
 """Summarizes text in one sentence (mock — just returns first sentence)."""
 sentences = text.split(".")
 return sentences[0].strip() + "." if sentences else text

@tool
def code_analyzer(code: str) -> str:
 """
 Analyzes Python code and returns basic stats.
 Reports: number of lines, functions defined, imports used.
 """
 lines = code.strip().split("\n")
 num_lines = len(lines)
 num_functions = sum(1 for l in lines if l.strip().startswith("def "))
 num_imports = sum(1 for l in lines if l.strip().startswith(("import ", "from ")))
 return (
 f"Code analysis: {num_lines} lines, "
 f"{num_functions} function(s), "
 f"{num_imports} import(s)"
 )

# Helper

def run_agent(graph, question: str, config: dict = None) -> str:
 """Run a LangGraph agent and return the final answer."""
 invoke_config = config or {}
 result = graph.invoke(
 {"messages": [HumanMessage(content=question)]},
 config=invoke_config,
 )
 return result["messages"][-1].content

# Feature 10A: Parallel Agents
def parallel_agents_demo():
 print("\n" + "="*60)
 print("FEATURE 10A: Parallel Agents")
 print("="*60)
 print("Multiple specialist agents work independently on different tasks.")
 print("Results are combined at the end.\n")

 llm = ChatOllama(model="llama3.2", temperature=0)

 # Agent 1: Math specialist
 math_agent = create_react_agent(llm, [calculate, get_current_time])

 # Agent 2: Text specialist
 text_agent = create_react_agent(llm, [word_counter, text_summarizer])

 # Agent 3: Code specialist
 code_agent = create_react_agent(llm, [code_analyzer, word_counter])

 tasks = {
 "Math Agent": "What is 256 divided by 16? Then multiply the result by 7.",
 "Text Agent": "How many words are in this text: 'LangChain is a powerful framework for building LLM applications with ease and flexibility'?",
 "Code Agent": "Analyze this code: import os\nimport sys\n\ndef hello():\n print('hello')\n\ndef world():\n print('world')",
 }

 results = {}
 for agent_name, task in tasks.items():
 agent = {"Math Agent": math_agent, "Text Agent": text_agent, "Code Agent": code_agent}[agent_name]
 print(f"--- {agent_name} ---")
 print(f"Task: {task[:80]}...")
 result = run_agent(agent, task)
 results[agent_name] = result
 print(f"Result: {result}\n")

 print("--- Combined Results ---")
 for agent, result in results.items():
 print(f" {agent}: {result[:80]}")

# Feature 10B: Sequential Agents (Pipeline)
def sequential_agents_demo():
 print("\n" + "="*60)
 print("FEATURE 10B: Sequential Agents (Pipeline)")
 print("="*60)
 print("Output of Agent 1 → feeds as input → Agent 2\n")

 llm = ChatOllama(model="llama3.2", temperature=0)

 # Agent 1: Analyzes the question and extracts the math part
 agent1 = create_react_agent(llm, [calculate])

 # Agent 2: Takes the math result and formats a final answer
 agent2 = create_react_agent(llm, [word_counter, get_current_time])

 original_task = "Calculate 15 times 8, then tell me how many digits the answer has."

 print(f"Original Task: {original_task}\n")

 # Step 1: Agent 1 solves the math
 print("Step 1 — Math Agent:")
 step1_result = run_agent(agent1, original_task)
 print(f" Output: {step1_result}\n")

 # Step 2: Agent 2 gets Agent 1's result as context
 step2_task = f"Previous agent calculated: '{step1_result}'. Now count how many words are in that answer."
 print("Step 2 — Text Agent (receives Step 1 output):")
 step2_result = run_agent(agent2, step2_task)
 print(f" Output: {step2_result}")

# Feature 10C: Supervisor Pattern
def supervisor_pattern_demo():
 print("\n" + "="*60)
 print("FEATURE 10C: Supervisor Pattern")
 print("="*60)
 print("One supervisor LLM reads the task and routes it to the right specialist.\n")

 llm = ChatOllama(model="llama3.2", temperature=0)

 # Specialized agents
 math_agent = create_react_agent(llm, [calculate])
 text_agent = create_react_agent(llm, [word_counter, text_summarizer])
 time_agent = create_react_agent(llm, [get_current_time])

 agents = {
 "math": math_agent,
 "text": text_agent,
 "time": time_agent,
 }

 def supervisor_route(task: str) -> str:
 """
 Supervisor: uses LLM to decide which agent should handle the task.
 Returns agent key: 'math', 'text', or 'time'.
 """
 routing_prompt = [
 SystemMessage(content=(
 "You are a router. Given a task, reply with ONLY one word:\n"
 "- 'math' if the task involves numbers or calculations\n"
 "- 'text' if the task involves counting words or summarizing\n"
 "- 'time' if the task involves current date or time\n"
 "Reply with ONLY the single word, nothing else."
 )),
 HumanMessage(content=f"Task: {task}"),
 ]
 response = llm.invoke(routing_prompt)
 route = response.content.strip().lower().split()[0]
 return route if route in agents else "math" # default fallback

 def supervised_pipeline(task: str) -> str:
 route = supervisor_route(task)
 print(f" Supervisor routed to: [{route} agent]")
 agent = agents[route]
 return run_agent(agent, task)

 tasks = [
 "What is 144 divided by 12?",
 "How many words are in: 'The quick brown fox jumps over the lazy dog'?",
 "What is today's date?",
 "Calculate 99 multiplied by 11.",
 ]

 for task in tasks:
 print(f"\nTask: {task}")
 result = supervised_pipeline(task)
 print(f"Result: {result}")

# Feature 10D: Agent with Shared Memory
def shared_memory_demo():
 print("\n" + "="*60)
 print("FEATURE 10D: Multiple Agents Sharing Memory")
 print("="*60)
 print("Two agents share the same thread_id — they see each other's work.\n")

 llm = ChatOllama(model="llama3.2", temperature=0)
 memory = MemorySaver()

 # Both agents use the SAME memory store but are specialized differently
 math_agent = create_react_agent(llm, [calculate, get_current_time], checkpointer=memory)
 text_agent = create_react_agent(llm, [word_counter], checkpointer=memory)

 # Shared session
 shared_config = {"configurable": {"thread_id": "shared-session-001"}}

 print("--- Math Agent turn ---")
 result1 = run_agent(math_agent, "Calculate 50 plus 25.", config=shared_config)
 print(f"Math Agent: {result1}\n")

 print("--- Text Agent turn (same session — sees math agent's work) ---")
 result2 = run_agent(text_agent, "What was the result of the last calculation done in this session?", config=shared_config)
 print(f"Text Agent: {result2}")

# Main
def main():
 if not check_ollama():
 print(" Ollama is not running! Start with: ollama serve")
 return

 print("\n Feature 10: Multi-Agent Patterns ( with Ollama + LangGraph)")
 print(" Using llama3.2.\n")

 parallel_agents_demo()
 sequential_agents_demo()
 supervisor_pattern_demo()
 shared_memory_demo()

 print("\n" + "="*60)
 print(" Multi-Agent examples completed!")
 print("="*60)
 print("\n Multi-agent patterns summary:")
 print(" Parallel → agents work independently, results combined")
 print(" Sequential → output of agent 1 becomes input of agent 2")
 print(" Supervisor → one LLM decides which specialist handles the task")
 print(" Shared Mem → agents share context via same thread_id")
 print("="*60)

if __name__ == "__main__":
 main()
