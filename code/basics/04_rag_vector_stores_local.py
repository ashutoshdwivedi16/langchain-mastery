"""
Feature 4: RAG (Retrieval-Augmented Generation) - LOCAL VERSION
Uses HuggingFace embeddings (free) and local LLM
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

def check_ollama():
    """Check if Ollama is accessible"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

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
                        "HuggingFace provides many free embedding models. Embeddings enable "
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
            page_content="Ollama allows you to run large language models locally on your computer. "
                        "It supports models like Llama 2, Mistral, and Phi. You can use Ollama with "
                        "LangChain without needing API keys or internet access.",
            metadata={"source": "ollama_guide", "topic": "local_llms"}
        ),
    ]
    return docs

def huggingface_embeddings_example():
    """Using free HuggingFace embeddings"""
    print("\n" + "="*60)
    print("FEATURE 4A: Free HuggingFace Embeddings")
    print("="*60)

    print("\n🚀 Loading HuggingFace embeddings model...")
    print("   (First time may take a few minutes to download)")

    # Use a small, efficient model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("✅ Embeddings model loaded!\n")

    # Test embeddings
    text = "This is a test sentence."
    embedding = embeddings.embed_query(text)

    print(f"Text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    return embeddings

def faiss_vector_store_example():
    """Using FAISS with free embeddings"""
    print("\n" + "="*60)
    print("FEATURE 4B: FAISS Vector Store (Free & Local)")
    print("="*60)

    print("\n📥 Creating embeddings and vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    docs = create_sample_documents()

    # Create FAISS vector store
    print("✅ Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Perform similarity search
    print("\n--- Similarity Search ---")
    query = "How do I store and search through documents?"
    results = vectorstore.similarity_search(query, k=2)

    print(f"\nQuery: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata['source']}\n")

    # Save the vector store
    vectorstore.save_local("/Users/ashutosh/Cloude/langchain-hello-world/faiss_index")
    print("💾 Vector store saved to: faiss_index/")

    return vectorstore

def rag_with_local_llm():
    """Complete RAG with local LLM and free embeddings"""
    print("\n" + "="*60)
    print("FEATURE 4C: RAG with Local LLM (100% Free!)")
    print("="*60)

    if not check_ollama():
        print("\n❌ Ollama is not running!")
        print("Please start Ollama to run this example.")
        return

    print("\n🔧 Setting up RAG system...")

    # Setup local LLM
    llm = Ollama(model="llama2", temperature=0)

    # Setup free embeddings
    print("📥 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create documents
    docs = create_sample_documents()

    # Create vector store
    print("📊 Creating vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create RAG chain
    print("⛓️  Creating RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

    print("✅ RAG system ready!\n")

    # Ask questions
    questions = [
        "What is LangChain?",
        "How does RAG work?",
        "What is Ollama and why would I use it?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print('='*60)

        result = qa_chain.invoke({"query": question})

        print(f"\n💬 Answer:\n{result['result']}")

        print(f"\n📚 Sources:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"  {i}. {doc.metadata['source']} - {doc.metadata['topic']}")

def text_splitting_example():
    """Demonstrates text splitting"""
    print("\n" + "="*60)
    print("FEATURE 4D: Text Splitting")
    print("="*60)

    long_text = """
    LangChain is a powerful framework for building applications with Large Language Models.
    It works great with both API-based models like OpenAI and local models like Ollama.

    One of the best things about LangChain is that you can build sophisticated AI applications
    completely for free using local models. Ollama provides easy access to models like Llama 2,
    Mistral, and Phi that run entirely on your computer.

    For embeddings, you can use HuggingFace's sentence-transformers which are completely free
    and work offline. This means you can build RAG systems without any API costs.

    Vector databases like FAISS and Chroma can run locally too, giving you full control over
    your data and zero ongoing costs. This makes LangChain perfect for learning, prototyping,
    and even production applications where you want to keep costs low.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        length_function=len
    )

    chunks = text_splitter.split_text(long_text)

    print(f"\nOriginal text length: {len(long_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"\n📄 Chunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk.strip())

def main():
    print("\n💡 This example uses 100% FREE local tools:")
    print("   - Ollama for LLM (free, local)")
    print("   - HuggingFace for embeddings (free, open-source)")
    print("   - FAISS for vector store (free, local)")
    print("   - No API keys needed!\n")

    huggingface_embeddings_example()
    faiss_vector_store_example()
    text_splitting_example()
    rag_with_local_llm()

    print("\n" + "="*60)
    print("✅ RAG examples completed!")
    print("="*60)
    print("\n💡 What you learned:")
    print("  ✓ Use free HuggingFace embeddings")
    print("  ✓ Build vector stores with FAISS")
    print("  ✓ Create RAG systems without APIs")
    print("  ✓ Split documents for better retrieval")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        import requests
        main()
    except ImportError:
        print("Please install: pip install requests")
