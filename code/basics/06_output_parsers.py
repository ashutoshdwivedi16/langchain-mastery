"""
Feature 6: Output Parsers and Structured Data
Demonstrates how to parse LLM outputs into structured formats
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from typing import List

load_dotenv()

def comma_separated_list_parser():
    """Parse output as a comma-separated list"""
    print("\n" + "="*60)
    print("FEATURE 6A: Comma-Separated List Parser")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create parser
    output_parser = CommaSeparatedListOutputParser()

    # Create prompt with format instructions
    prompt = PromptTemplate(
        template="List 5 {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    # Create chain
    chain = prompt | llm | output_parser

    # Execute
    result = chain.invoke({"subject": "popular programming languages"})

    print(f"\nRaw result type: {type(result)}")
    print(f"Parsed result: {result}")
    print(f"\nFirst item: {result[0]}")
    print(f"Number of items: {len(result)}")

def structured_output_parser():
    """Parse output into a structured dictionary"""
    print("\n" + "="*60)
    print("FEATURE 6B: Structured Output Parser")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Define response schema
    response_schemas = [
        ResponseSchema(name="name", description="The name of the person"),
        ResponseSchema(name="age", description="The age of the person"),
        ResponseSchema(name="occupation", description="The occupation of the person"),
        ResponseSchema(name="hobbies", description="A comma-separated list of hobbies")
    ]

    # Create parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Create prompt
    prompt = PromptTemplate(
        template="Generate a fictional character profile.\n{format_instructions}\n{query}",
        input_variables=["query"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    # Create chain
    chain = prompt | llm | output_parser

    # Execute
    result = chain.invoke({"query": "Create a profile for a space explorer."})

    print(f"\nResult type: {type(result)}")
    print(f"Parsed result: {result}")
    print(f"\nName: {result['name']}")
    print(f"Age: {result['age']}")
    print(f"Occupation: {result['occupation']}")
    print(f"Hobbies: {result['hobbies']}")

def pydantic_output_parser():
    """Parse output into Pydantic models for type safety"""
    print("\n" + "="*60)
    print("FEATURE 6C: Pydantic Output Parser")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Define Pydantic model
    class Book(BaseModel):
        title: str = Field(description="The title of the book")
        author: str = Field(description="The author of the book")
        year: int = Field(description="The year the book was published")
        genre: str = Field(description="The genre of the book")
        rating: float = Field(description="Rating out of 5.0", ge=0, le=5)

        @validator('year')
        def year_must_be_reasonable(cls, v):
            if v < 1000 or v > 2025:
                raise ValueError('Year must be between 1000 and 2025')
            return v

    # Create parser
    output_parser = PydanticOutputParser(pydantic_object=Book)

    # Create prompt
    prompt = PromptTemplate(
        template="Generate information about a famous book.\n{format_instructions}\n{query}",
        input_variables=["query"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    # Create chain
    chain = prompt | llm | output_parser

    # Execute
    result = chain.invoke({"query": "Tell me about a classic science fiction novel."})

    print(f"\nResult type: {type(result)}")
    print(f"Parsed result: {result}")
    print(f"\nTitle: {result.title}")
    print(f"Author: {result.author}")
    print(f"Year: {result.year}")
    print(f"Genre: {result.genre}")
    print(f"Rating: {result.rating}/5.0")

def complex_pydantic_parser():
    """Parse nested and complex structures"""
    print("\n" + "="*60)
    print("FEATURE 6D: Complex Pydantic Structures")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Define nested Pydantic models
    class Address(BaseModel):
        street: str = Field(description="Street address")
        city: str = Field(description="City name")
        country: str = Field(description="Country name")

    class Person(BaseModel):
        name: str = Field(description="Full name of the person")
        age: int = Field(description="Age in years", ge=0, le=150)
        email: str = Field(description="Email address")
        address: Address = Field(description="Residential address")
        skills: List[str] = Field(description="List of professional skills")

    # Create parser
    output_parser = PydanticOutputParser(pydantic_object=Person)

    # Create prompt
    prompt = PromptTemplate(
        template="Generate a professional profile for a software engineer.\n{format_instructions}",
        input_variables=[],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    # Create chain
    chain = prompt | llm | output_parser

    # Execute
    result = chain.invoke({})

    print(f"\nResult type: {type(result)}")
    print(f"\n{'='*60}")
    print("Parsed Professional Profile:")
    print('='*60)
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Email: {result.email}")
    print(f"\nAddress:")
    print(f"  Street: {result.address.street}")
    print(f"  City: {result.address.city}")
    print(f"  Country: {result.address.country}")
    print(f"\nSkills:")
    for skill in result.skills:
        print(f"  - {skill}")

def json_output_example():
    """Getting JSON output directly"""
    print("\n" + "="*60)
    print("FEATURE 6E: JSON Output Mode")
    print("="*60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    prompt = PromptTemplate(
        template="""Generate a recipe in JSON format with the following fields:
        - name: recipe name
        - ingredients: list of ingredients
        - steps: list of cooking steps
        - prep_time: preparation time in minutes
        - difficulty: easy, medium, or hard

        Recipe type: {recipe_type}

        Return only valid JSON, no other text.""",
        input_variables=["recipe_type"]
    )

    chain = prompt | llm

    result = chain.invoke({"recipe_type": "chocolate cake"})

    print(f"\nRaw output:\n{result.content}")

    # Parse JSON manually
    import json
    try:
        parsed = json.loads(result.content)
        print(f"\n{'='*60}")
        print("Parsed Recipe:")
        print('='*60)
        print(f"Name: {parsed['name']}")
        print(f"Prep Time: {parsed['prep_time']} minutes")
        print(f"Difficulty: {parsed['difficulty']}")
        print(f"\nIngredients:")
        for ingredient in parsed['ingredients']:
            print(f"  - {ingredient}")
        print(f"\nSteps:")
        for i, step in enumerate(parsed['steps'], 1):
            print(f"  {i}. {step}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

def main():
    comma_separated_list_parser()
    structured_output_parser()
    pydantic_output_parser()
    complex_pydantic_parser()
    json_output_example()

if __name__ == "__main__":
    main()
