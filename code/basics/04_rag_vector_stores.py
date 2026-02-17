"""
Feature 4: RAG (Retrieval-Augmented Generation) with Vector Stores
Demonstrates document loading, text splitting, embeddings, and semantic search
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

load_dotenv()

def create_sample_documents():
    """Create sample documents for RAG demo"""
    docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
                        "It provides tools for prompt management, chains, data augmented generation, agents, "
                        "and memory management.",
            metadata={"source": "langchain_intro", "topic": "basics"}
        ),
        Document(
            page_content="Vector databases store embeddings and enable semantic search. Popular vector databases "
                        "include Chroma, Pinecone, FAISS, and Weaviate. They allow you to find similar documents "
                        "based on meaning rather than exact keyword matches.",
            metadata={"source": "vector_db_guide", "topic": "databases"}
        ),
        Document(
            page_content="Embeddings are numerical representations of text that capture semantic meaning. "
                        "OpenAI's text-embedding-ada-002 is a popular embedding model. Embeddings enable "
                        "similarity search and are fundamental to RAG applications.",
            metadata={"source": "embeddings_explained", "topic": "embeddings"}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines retrieval with generation. "
                        "First, relevant documents are retrieved from a knowledge base. Then, these documents "
                        "are provided as context to an LLM to generate accurate, grounded responses.",
            metadata={"source": "rag_overview", "topic": "rag"}
        ),
        Document(
            page_content="Agents in LangChain can use tools to interact with external systems. "
                        "They can decide which tools to use and in what order based on the user's input. "
                        "Common tools include calculators, search engines, and databases.",
            metadata={"source": "agents_guide", "topic": "agents"}
        ),
    ]
    return docs

def chroma_vector_store_example():
    """Using Chroma vector store for semantic search"""
    print("\n" + "="*60)
    print("FEATURE 4A: Chroma Vector Store")
    print("="*60)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create sample documents
    docs = create_sample_documents()

    # Create vector store
    print("\nCreating Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="langchain_hello_world"
    )

    # Perform similarity search
    print("\n--- Similarity Search ---")
    query = "How do I store and search through documents?"
    results = vectorstore.similarity_search(query, k=2)

    print(f"Query: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}\n")

    # Similarity search with scores
    print("\n--- Similarity Search with Scores ---")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"Result {i} (Score: {score:.4f}):")
        print(f"Content: {doc.page_content[:100]}...\n")

    return vectorstore

def faiss_vector_store_example():
    """Using FAISS vector store (faster for large datasets)"""
    print("\n" + "="*60)
    print("FEATURE 4B: FAISS Vector Store")
    print("="*60)

    embeddings = OpenAIEmbeddings()
    docs = create_sample_documents()

    # Create FAISS vector store
    print("\nCreating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Perform search
    query = "What are agents?"
    results = vectorstore.similarity_search(query, k=1)

    print(f"\nQuery: {query}")
    print(f"Result: {results[0].page_content}")

    return vectorstore

def rag_chain_example():
    """Complete RAG chain with question answering"""
    print("\n" + "="*60)
    print("FEATURE 4C: RAG Question Answering Chain")
    print("="*60)

    # Setup
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings()
    docs = create_sample_documents()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_qa_demo"
    )

    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

    # Ask questions
    questions = [
        "What is LangChain and what does it provide?",
        "Explain how RAG works.",
        "What are the benefits of using embeddings?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

        result = qa_chain.invoke({"query": question})

        print(f"\nAnswer: {result['result']}")

        print(f"\nSources:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"  {i}. {doc.metadata['source']} (Topic: {doc.metadata['topic']})")

def text_splitting_example():
    """Demonstrates text splitting for large documents"""
    print("\n" + "="*60)
    print("FEATURE 4D: Text Splitting")
    print("="*60)

    # Long text to split
    long_text = """
    LangChain is a powerful framework for building applications with Large Language Models (LLMs).
    It provides a standard interface for chains, lots of integrations with other tools, and end-to-end
    chains for common applications.

    One of the key components of LangChain is its support for Retrieval-Augmented Generation (RAG).
    RAG is a technique that combines the power of retrieval systems with generative models to produce
    more accurate and contextually relevant responses.

    The framework also includes support for agents, which are entities that can use tools to interact
    with the outside world. Agents can make decisions about which tools to use and in what order based
    on user input and intermediate results.

    Memory is another important feature, allowing chains and agents to maintain context across multiple
    interactions. This is crucial for building conversational AI applications that need to remember
    previous parts of the conversation.
    """

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )

    # Split text
    chunks = text_splitter.split_text(long_text)

    print(f"\nOriginal text length: {len(long_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"\nChunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk.strip())

def main():
    chroma_vector_store_example()
    faiss_vector_store_example()
    rag_chain_example()
    text_splitting_example()

if __name__ == "__main__":
    main()
