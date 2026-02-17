"""
Feature 7: Advanced Semantic Search
Demonstrates similarity search with relevance scores, MultiQueryRetriever,
ContextualCompressionRetriever, and MMR-based RAG pipelines.
"""

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_ollama():
 import requests
 try:
 return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
 except:
 return False

# Shared knowledge base

DOCUMENTS = [
 Document(page_content="Python is a high-level programming language known for simplicity and readability. Created by Guido van Rossum in 1991.", metadata={"source": "python_intro", "topic": "python", "year": 1991}),
 Document(page_content="Python is widely used in data science, machine learning, web development, and automation scripting.", metadata={"source": "python_uses", "topic": "python", "year": 2024}),
 Document(page_content="LangChain is a framework for building LLM-powered applications. It provides tools for chaining prompts, memory, agents, and retrievers.", metadata={"source": "langchain_intro", "topic": "langchain", "year": 2023}),
 Document(page_content="LangChain supports both local models via Ollama and cloud models via OpenAI, Anthropic, and others.", metadata={"source": "langchain_models", "topic": "langchain", "year": 2023}),
 Document(page_content="RAG (Retrieval-Augmented Generation) improves LLM accuracy by fetching relevant documents before generating a response.", metadata={"source": "rag_intro", "topic": "rag", "year": 2023}),
 Document(page_content="FAISS is a library for efficient similarity search developed by Facebook AI. It stores vectors and finds nearest neighbours.", metadata={"source": "faiss_intro", "topic": "vector_db", "year": 2019}),
 Document(page_content="Ollama allows you to run large language models locally on your machine.", metadata={"source": "ollama_intro", "topic": "ollama", "year": 2023}),
 Document(page_content="Machine learning is a subset of AI where systems learn patterns from data without being explicitly programmed.", metadata={"source": "ml_intro", "topic": "ml", "year": 2024}),
 Document(page_content="Neural networks are computational models inspired by the human brain, consisting of layers of interconnected nodes.", metadata={"source": "nn_intro", "topic": "ml", "year": 2024}),
 Document(page_content="Transformers are a neural network architecture introduced in 2017 that revolutionised NLP by using self-attention mechanisms.", metadata={"source": "transformers_intro", "topic": "ml", "year": 2017}),
]

def build_vectorstore(embeddings):
 """Build FAISS vectorstore from shared documents."""
 return FAISS.from_documents(DOCUMENTS, embeddings)

def get_embeddings():
 print(" Loading HuggingFace embeddings model...")
 return HuggingFaceEmbeddings(
 model_name="sentence-transformers/all-MiniLM-L6-v2",
 model_kwargs={"device": "cpu"},
 )

# Feature 7A: Basic Similarity Search
def basic_similarity_search(vectorstore):
 print("\n" + "="*60)
 print("FEATURE 7A: Basic Similarity Search")
 print("="*60)
 print("Finds documents whose MEANING is closest to the query.\n")

 queries = [
 "How do I run AI models locally?",
 "What is used for finding similar vectors?",
 ]

 retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

 for query in queries:
 print(f"Query: {query}")
 docs = retriever.invoke(query)
 for i, doc in enumerate(docs, 1):
 print(f" Result {i}: [{doc.metadata['source']}] {doc.page_content[:80]}...")
 print()

# Feature 7B: Similarity Score Search
def similarity_score_search(vectorstore):
 print("\n" + "="*60)
 print("FEATURE 7B: Similarity Search with Relevance Scores")
 print("="*60)
 print("Shows HOW similar each result is (score closer to 1.0 = more relevant).\n")

 query = "local LLM inference"
 results = vectorstore.similarity_search_with_relevance_scores(query, k=4)

 print(f"Query: '{query}'\n")
 for doc, score in results:
 bar = "" * int(score * 20)
 print(f" Score: {score:.3f} {bar}")
 print(f" Source: [{doc.metadata['source']}]")
 print(f" Content: {doc.page_content[:70]}...")
 print()

# Feature 7C: MultiQueryRetriever
def multi_query_retriever(vectorstore, llm):
 print("\n" + "="*60)
 print("FEATURE 7C: MultiQueryRetriever")
 print("="*60)
 print("Uses LLM to generate MULTIPLE versions of your query.")
 print("Retrieves more diverse, relevant documents.\n")

 from langchain_classic.retrievers.multi_query import MultiQueryRetriever

 base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

 # MultiQueryRetriever uses the LLM to rephrase the query 3 ways
 multi_retriever = MultiQueryRetriever.from_llm(
 retriever=base_retriever,
 llm=llm,
 )

 query = "How can I build AI applications without paying for APIs?"
 print(f"Original query: '{query}'\n")
 print("(LLM generates 3 variations of this query internally...)\n")

 docs = multi_retriever.invoke(query)
 # Deduplicate
 seen = set()
 unique_docs = []
 for doc in docs:
 if doc.page_content not in seen:
 seen.add(doc.page_content)
 unique_docs.append(doc)

 print(f"Retrieved {len(unique_docs)} unique documents:\n")
 for i, doc in enumerate(unique_docs, 1):
 print(f" {i}. [{doc.metadata['source']}] {doc.page_content[:80]}...")

# Feature 7D: Contextual Compression Retriever
def contextual_compression_retriever(vectorstore, llm):
 print("\n" + "="*60)
 print("FEATURE 7D: Contextual Compression Retriever")
 print("="*60)
 print("Retrieves docs, then COMPRESSES each one down to only")
 print("the parts relevant to your query — reduces noise.\n")

 from langchain_classic.retrievers import ContextualCompressionRetriever
 from langchain_classic.retrievers.document_compressors import LLMChainExtractor

 base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

 # LLMChainExtractor: extracts only relevant sentences from each doc
 compressor = LLMChainExtractor.from_llm(llm)

 compression_retriever = ContextualCompressionRetriever(
 base_compressor=compressor,
 base_retriever=base_retriever,
 )

 query = "What framework helps build LLM apps?"
 print(f"Query: '{query}'\n")

 # Show uncompressed vs compressed
 print("--- Without compression (full docs) ---")
 raw_docs = base_retriever.invoke(query)
 for doc in raw_docs:
 print(f" [{doc.metadata['source']}]: {doc.page_content}")

 print("\n--- With compression (only relevant parts) ---")
 try:
 compressed_docs = compression_retriever.invoke(query)
 if compressed_docs:
 for doc in compressed_docs:
 print(f" [{doc.metadata['source']}]: {doc.page_content}")
 else:
 print(" (All content compressed away — query too specific)")
 except Exception as e:
 print(f" Compression error: {e}")

# Feature 7E: Full RAG pipeline with advanced retriever
def advanced_rag_pipeline(vectorstore, llm):
 print("\n" + "="*60)
 print("FEATURE 7E: Full RAG Pipeline with Semantic Search")
 print("="*60)

 retriever = vectorstore.as_retriever(
 search_type="mmr", # Maximum Marginal Relevance
 search_kwargs={"k": 3, "fetch_k": 6}, # fetch 6, return top 3 diverse
 )

 # MMR = picks results that are relevant BUT also diverse from each other
 # avoids returning 3 very similar documents

 rag_prompt = ChatPromptTemplate.from_messages([
 ("system",
 "Answer the question using ONLY the context below. "
 "If the answer is not in the context, say 'I don't have that information'.\n\n"
 "Context:\n{context}"),
 ("human", "{question}"),
 ])

 def format_docs(docs):
 return "\n\n".join(
 f"[{d.metadata['source']}]: {d.page_content}" for d in docs
 )

 rag_chain = (
 {"context": retriever | format_docs, "question": RunnablePassthrough()}
 | rag_prompt
 | llm
 | StrOutputParser()
 )

 questions = [
 "What is LangChain and what can I build with it?",
 "How can I run LLMs locally?",
 "What is RAG and why is it useful?",
 ]

 for q in questions:
 print(f"\nQ: {q}")
 print("-"*40)
 answer = rag_chain.invoke(q)
 print(f"A: {answer}")

# Main
def main():
 if not check_ollama():
 print(" Ollama is not running! Start with: ollama serve")
 return

 print("\n Feature 7: Advanced Semantic Search (!)")

 embeddings = get_embeddings()
 vectorstore = build_vectorstore(embeddings)
 llm = OllamaLLM(model="llama2", temperature=0)

 print(" Vectorstore built with", len(DOCUMENTS), "documents")

 basic_similarity_search(vectorstore)
 similarity_score_search(vectorstore)
 multi_query_retriever(vectorstore, llm)
 contextual_compression_retriever(vectorstore, llm)
 advanced_rag_pipeline(vectorstore, llm)

 print("\n" + "="*60)
 print(" Semantic Search examples completed!")
 print("="*60)

if __name__ == "__main__":
 main()
