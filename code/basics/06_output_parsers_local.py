"""
Feature 6: Output Parsers - LOCAL VERSION (No API Required)
Uses Ollama for local LLM inference.
Demonstrates StrOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field, field_validator
from typing import List


def check_ollama():
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# ── Feature 6A: StrOutputParser ───────────────────────────────────────────
def str_output_parser_example():
    print("\n" + "="*60)
    print("FEATURE 6A: StrOutputParser (plain text)")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.5)
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Give me one interesting fact about {topic} in a single sentence."
    )

    chain = prompt | llm | StrOutputParser()

    for topic in ["Python", "LangChain", "Machine Learning"]:
        result = chain.invoke({"topic": topic})
        print(f"\n{topic}: {result}")


# ── Feature 6B: CommaSeparatedListOutputParser ────────────────────────────
def list_output_parser_example():
    print("\n" + "="*60)
    print("FEATURE 6B: CommaSeparatedListOutputParser")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.3)
    parser = CommaSeparatedListOutputParser()

    prompt = PromptTemplate(
        input_variables=["subject", "n"],
        template=(
            "List exactly {n} popular {subject}. "
            "Respond with ONLY the names separated by commas, no numbering, no extra text.\n"
            "{format_instructions}"
        ),
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    examples = [
        {"subject": "Python libraries for data science", "n": "5"},
        {"subject": "programming languages",             "n": "4"},
    ]

    for ex in examples:
        print(f"\nSubject: {ex['subject']}")
        result = chain.invoke(ex)
        print(f"Parsed list ({len(result)} items): {result}")


# ── Feature 6C: PydanticOutputParser ──────────────────────────────────────
class BookInfo(BaseModel):
    title:  str          = Field(description="Title of the book")
    author: str          = Field(description="Author of the book")
    year:   int          = Field(description="Year the book was published")
    genre:  str          = Field(description="Genre of the book")
    summary: str         = Field(description="One-sentence summary of the book")

    @field_validator("year")
    @classmethod
    def year_must_be_valid(cls, v: int) -> int:
        if v < 1000 or v > 2026:
            raise ValueError("Year must be between 1000 and 2026")
        return v


def pydantic_output_parser_example():
    print("\n" + "="*60)
    print("FEATURE 6C: PydanticOutputParser (structured JSON → object)")
    print("="*60)

    # llama3.2 follows JSON instructions better than llama2
    llm = OllamaLLM(model="llama3.2", temperature=0)
    parser = PydanticOutputParser(pydantic_object=BookInfo)

    prompt = PromptTemplate(
        input_variables=["book_title"],
        template=(
            "You are a helpful assistant. Return ONLY a JSON object with these exact fields for the book '{book_title}':\n"
            "title, author, year (integer), genre, summary (one sentence).\n\n"
            "Example format:\n"
            '{{"title": "Book Name", "author": "Author Name", "year": 2000, "genre": "Fiction", "summary": "A story about..."}}\n\n'
            "Now provide the JSON for '{book_title}'. Return ONLY the JSON, nothing else."
        ),
    )

    chain = prompt | llm | parser

    books = ["The Pragmatic Programmer", "Clean Code"]

    for book in books:
        print(f"\nBook: {book}")
        print("-"*40)
        try:
            result = chain.invoke({"book_title": book})
            print(f"Title   : {result.title}")
            print(f"Author  : {result.author}")
            print(f"Year    : {result.year}")
            print(f"Genre   : {result.genre}")
            print(f"Summary : {result.summary}")
        except Exception as e:
            print(f"Parse error (LLM didn't follow format exactly): {e}")


# ── Feature 6D: StructuredOutputParser ────────────────────────────────────
def structured_output_parser_example():
    print("\n" + "="*60)
    print("FEATURE 6D: StructuredOutputParser (schema-defined)")
    print("="*60)

    llm = OllamaLLM(model="llama2", temperature=0.3)

    response_schemas = [
        ResponseSchema(name="language",    description="The programming language name"),
        ResponseSchema(name="created_year",description="The year the language was created"),
        ResponseSchema(name="primary_use", description="What the language is primarily used for"),
        ResponseSchema(name="fun_fact",    description="One interesting or surprising fact about it"),
    ]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = PromptTemplate(
        input_variables=["language"],
        template=(
            "Provide information about the {language} programming language.\n"
            "{format_instructions}\n"
            "Respond with ONLY the markdown JSON block."
        ),
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    for lang in ["Python", "JavaScript"]:
        print(f"\nLanguage: {lang}")
        print("-"*40)
        try:
            result = chain.invoke({"language": lang})
            for k, v in result.items():
                print(f"  {k:15s}: {v}")
        except Exception as e:
            print(f"Parse error: {e}")


def main():
    if not check_ollama():
        print("\n❌ Ollama is not running!")
        print("Start it with: ollama serve")
        return

    print("\n💡 Feature 6: Output Parsers (100% Local, No API needed!)")

    str_output_parser_example()
    list_output_parser_example()
    pydantic_output_parser_example()
    structured_output_parser_example()

    print("\n" + "="*60)
    print("✅ Output Parsers examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
