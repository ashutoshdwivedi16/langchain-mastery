"""
Feature 2: Prompt Templates and Chains
Demonstrates prompt engineering with templates and chaining operations
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def prompt_template_example():
    """Using prompt templates for reusable prompts"""
    print("\n" + "="*60)
    print("FEATURE 2A: Prompt Templates")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Simple prompt template
    template = """You are a helpful assistant that translates {input_language} to {output_language}.

    Text to translate: {text}

    Translation:"""

    prompt = PromptTemplate(
        input_variables=["input_language", "output_language", "text"],
        template=template
    )

    # Create a chain using LCEL (LangChain Expression Language)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "input_language": "English",
        "output_language": "Spanish",
        "text": "Hello World! How are you today?"
    })

    print(f"\nTranslation Result: {result}")

def chat_prompt_example():
    """Using chat prompt templates with system/human messages"""
    print("\n" + "="*60)
    print("FEATURE 2B: Chat Prompt Templates")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative poet who writes in the style of {style}."),
        ("human", "Write a short poem about {topic}.")
    ])

    chain = chat_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "style": "Shakespeare",
        "topic": "artificial intelligence"
    })

    print(f"\nPoem:\n{result}")

def sequential_chain_example():
    """Chaining multiple operations together"""
    print("\n" + "="*60)
    print("FEATURE 2C: Sequential Chains")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # First chain: Generate a company name
    prompt1 = ChatPromptTemplate.from_template(
        "Generate a creative company name for a business that sells {product}. "
        "Only return the company name, nothing else."
    )
    chain1 = prompt1 | llm | StrOutputParser()

    # Second chain: Generate a slogan
    prompt2 = ChatPromptTemplate.from_template(
        "Create a catchy slogan for a company called {company_name}. "
        "Only return the slogan, nothing else."
    )
    chain2 = prompt2 | llm | StrOutputParser()

    # Execute chains sequentially
    product = "eco-friendly water bottles"
    company_name = chain1.invoke({"product": product})
    slogan = chain2.invoke({"company_name": company_name})

    print(f"\nProduct: {product}")
    print(f"Company Name: {company_name}")
    print(f"Slogan: {slogan}")

def main():
    prompt_template_example()
    chat_prompt_example()
    sequential_chain_example()

if __name__ == "__main__":
    main()
