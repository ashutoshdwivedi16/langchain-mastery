"""
Feature 2: Prompts and Chains - LOCAL VERSION (No API Required)
Uses Ollama for local LLM inference.
Demonstrates PromptTemplates, ChatPromptTemplates, and LCEL chains.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def check_ollama():
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def prompt_template_example():
    """Basic PromptTemplate with variable substitution"""
    print("\n" + "="*60)
    print("FEATURE 2A: Basic PromptTemplate")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.7)

    # Simple prompt template
    template = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in exactly 2 sentences, simply and clearly."
    )

    chain = template | llm | StrOutputParser()

    topics = ["machine learning", "neural networks", "LangChain"]

    for topic in topics:
        print(f"\nTopic: {topic}")
        print("-"*40)
        response = chain.invoke({"topic": topic})
        print(response)


def multi_variable_prompt():
    """PromptTemplate with multiple variables"""
    print("\n" + "="*60)
    print("FEATURE 2B: Multi-Variable Prompt")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.7)

    template = PromptTemplate(
        input_variables=["language", "concept", "audience"],
        template=(
            "Explain the concept of {concept} in {language} programming language. "
            "The audience is {audience}. Keep it to 3 sentences."
        )
    )

    chain = template | llm | StrOutputParser()

    examples = [
        {"language": "Python", "concept": "list comprehension", "audience": "beginners"},
        {"language": "Python", "concept": "decorators",         "audience": "intermediate developers"},
    ]

    for ex in examples:
        print(f"\nLanguage: {ex['language']} | Concept: {ex['concept']} | Audience: {ex['audience']}")
        print("-"*40)
        print(chain.invoke(ex))


def chat_prompt_template_example():
    """ChatPromptTemplate with system + human messages"""
    print("\n" + "="*60)
    print("FEATURE 2C: ChatPromptTemplate")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.7)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {role}. Give clear, concise answers in 2-3 sentences."),
        ("human",  "{question}"),
    ])

    chain = chat_prompt | llm | StrOutputParser()

    examples = [
        {"role": "Python developer",    "question": "What is a decorator in Python?"},
        {"role": "machine learning engineer", "question": "What is overfitting?"},
    ]

    for ex in examples:
        print(f"\nRole: {ex['role']}")
        print(f"Question: {ex['question']}")
        print("-"*40)
        print(chain.invoke(ex))


def sequential_chain_example():
    """Two chains piped together — output of first feeds into second"""
    print("\n" + "="*60)
    print("FEATURE 2D: Sequential Chain (LCEL Pipe)")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.7)

    # Step 1: generate a concept explanation
    step1_prompt = PromptTemplate(
        input_variables=["concept"],
        template="Explain {concept} in one simple sentence."
    )

    # Step 2: turn that explanation into an analogy
    step2_prompt = PromptTemplate(
        input_variables=["explanation"],
        template="Take this explanation: '{explanation}'\nNow give a real-world analogy for it in one sentence."
    )

    # Chain: concept → explanation → analogy
    chain = (
        step1_prompt
        | llm
        | StrOutputParser()
        | (lambda explanation: {"explanation": explanation})
        | step2_prompt
        | llm
        | StrOutputParser()
    )

    concept = "recursion in programming"
    print(f"\nConcept: {concept}")
    print("-"*40)
    result = chain.invoke({"concept": concept})
    print(f"Analogy: {result}")


def main():
    if not check_ollama():
        print("\n❌ Ollama is not running!")
        print("Start it with: ollama serve")
        return

    print("\n💡 Feature 2: Prompts and Chains (100% Local, No API needed!)")

    prompt_template_example()
    multi_variable_prompt()
    chat_prompt_template_example()
    sequential_chain_example()

    print("\n" + "="*60)
    print("✅ Prompts and Chains examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
