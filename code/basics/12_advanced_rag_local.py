"""
Feature 12: Advanced RAG Strategies

Demonstrates production-grade RAG techniques that significantly improve accuracy:
- Query transformation (HyDE, multi-query, step-back)
- Reranking (cross-encoder after initial retrieval)
- Fusion retrieval (combine multiple strategies)
- Self-RAG (critique and refine)
- Contextual compression (filter irrelevant content)

These techniques address common RAG failure modes:
1. Query-document mismatch (user query != how info is phrased in docs)
2. Retrieval noise (top-k includes irrelevant chunks)
3. Missing context (answer requires multiple chunks)
4. Hallucination (LLM makes up facts not in retrieved context)

Prerequisites:
    pip install langchain langchain-community langchain-ollama
    pip install langchain-huggingface sentence-transformers
    pip install rank-bm25  # for BM25 retrieval
"""

from langchain_ollama import ChatOllama, OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi


# --- Knowledge Base: Software Architecture Documentation ---
KNOWLEDGE_BASE = [
    "The API Gateway uses JWT tokens for authentication. Tokens expire after 1 hour and must be refreshed using a refresh token that expires after 30 days.",
    "Rate limiting is implemented using a sliding window algorithm in Redis. The default limit is 1000 requests per minute per API key.",
    "The system uses PostgreSQL for transactional data with master-replica replication. Write operations go to the master, read operations are load-balanced across 3 replicas.",
    "Event sourcing is used for payment transactions. Every state change is recorded as an immutable event in Apache Kafka with 30 partitions.",
    "PCI-DSS Level 1 compliance is maintained. Card data is tokenized immediately at point of entry using a third-party tokenization service. No plaintext card numbers are stored.",
    "The fraud detection system uses an XGBoost classifier trained on 50+ features including transaction amount, merchant category, user location, device fingerprint, and time of day.",
    "High-risk transactions trigger step-up authentication. Users must verify via SMS OTP or biometric authentication before the transaction is approved.",
    "The system achieves 99.99% uptime through multi-region deployment across us-east-1, eu-west-1, and ap-southeast-1. Each region can handle 100% of traffic independently.",
    "Database backups run daily with full backups to S3 and 7-year retention for compliance. Point-in-time recovery is available for the past 30 days.",
    "API response times are monitored with a P95 latency target of 200ms. Alerts fire if P95 exceeds 250ms for 5 consecutive minutes.",
    "The caching layer uses a three-tier strategy: L1 application cache (Caffeine), L2 distributed cache (Redis), and L3 CDN edge cache for static content.",
    "Payment processing follows the flow: INITIATED → VALIDATED → AUTHORIZED → CAPTURED → SETTLED. Each state transition publishes an event to Kafka.",
    "The system handles 5 million transactions daily during normal operation and has scaled to 15 million during Black Friday traffic spikes.",
    "Security audits are performed quarterly by external assessors. All findings are tracked in Jira and must be resolved within 30 days for critical issues.",
    "Deployment uses blue-green strategy with automated rollback if error rate exceeds 1% or P95 latency exceeds 500ms within 5 minutes of deployment.",
]

# Global state for demos
llm = None
embeddings = None
vectorstore = None


def setup():
    """Initialize LLM, embeddings, and vector store"""
    global llm, embeddings, vectorstore

    print("Setting up RAG system...")
    llm = ChatOllama(model="llama3.2", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store from knowledge base
    docs = [Document(page_content=text, metadata={"source": f"doc_{i}"})
            for i, text in enumerate(KNOWLEDGE_BASE)]
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"✓ Loaded {len(KNOWLEDGE_BASE)} documents into vector store\n")


# --- 12A: Baseline RAG (No Enhancements) ---
def baseline_rag(query: str) -> str:
    """
    Baseline RAG: simple retrieval + generation.
    PROBLEM: Query-document mismatch can cause poor retrieval.
    """
    print("\n--- 12A: Baseline RAG ---")
    print(f"Query: '{query}'")

    # Retrieve top 3 chunks
    docs = vectorstore.similarity_search(query, k=3)

    # Generate answer
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    answer = llm.invoke(prompt).content
    print(f"\nRetrieved chunks:")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc.page_content[:100]}...")
    print(f"\nAnswer: {answer}")

    return answer


# --- 12B: HyDE (Hypothetical Document Embeddings) ---
def hyde_rag(query: str) -> str:
    """
    HyDE: Generate a hypothetical answer first, embed it, use it for retrieval.

    WHY: User query "How does auth work?" is short and vague.
    Hypothetical answer will use technical terms (JWT, tokens, expiration)
    that better match the actual documentation.
    """
    print("\n--- 12B: HyDE (Hypothetical Document Embeddings) ---")
    print(f"Query: '{query}'")

    # Step 1: Generate hypothetical answer (without seeing any docs)
    hyde_prompt = f"""Write a detailed, technical answer to this question as if you were
writing documentation. Use specific technical terms and implementation details.

Question: {query}

Hypothetical Answer:"""

    hypothetical_answer = llm.invoke(hyde_prompt).content
    print(f"\nHypothetical answer (for embedding):\n{hypothetical_answer[:200]}...\n")

    # Step 2: Embed the hypothetical answer and use it for retrieval
    hyde_embedding = embeddings.embed_query(hypothetical_answer)
    docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=3)

    # Step 3: Generate final answer using retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.invoke(final_prompt).content

    print(f"Retrieved chunks (via HyDE):")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc.page_content[:100]}...")
    print(f"\nFinal Answer: {answer}")

    return answer


# --- 12C: Multi-Query Retrieval ---
def multi_query_rag(query: str) -> str:
    """
    Multi-Query: Generate multiple variations of the query, retrieve for each, combine results.

    WHY: Single query might miss relevant docs due to wording.
    Multiple perspectives increase recall.
    """
    print("\n--- 12C: Multi-Query Retrieval ---")
    print(f"Original query: '{query}'")

    # Step 1: Generate query variations
    multi_query_prompt = f"""Generate 3 alternative versions of this question that ask for the same information
but use different wording and perspectives. Output only the questions, one per line.

Original question: {query}

Alternative questions:"""

    variations_text = llm.invoke(multi_query_prompt).content
    variations = [line.strip() for line in variations_text.split('\n') if line.strip() and not line.startswith('Alternative')]
    variations = variations[:3]  # Limit to 3

    print(f"\nQuery variations:")
    for i, var in enumerate(variations):
        print(f"  {i+1}. {var}")

    # Step 2: Retrieve for each variation
    all_docs = []
    seen_content = set()

    for var in variations:
        docs = vectorstore.similarity_search(var, k=2)
        for doc in docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

    print(f"\nTotal unique chunks retrieved: {len(all_docs)}")

    # Step 3: Generate answer from combined results
    context = "\n\n".join([doc.page_content for doc in all_docs[:5]])  # Top 5
    prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.invoke(prompt).content

    print(f"\nAnswer: {answer}")
    return answer


# --- 12D: Fusion Retrieval (Vector + BM25) ---
def fusion_retrieval(query: str) -> str:
    """
    Fusion: Combine vector similarity search with BM25 keyword search.

    WHY: Vector search handles semantic similarity.
         BM25 handles exact keyword matches.
         Combining both improves recall.
    """
    print("\n--- 12D: Fusion Retrieval (Vector + BM25) ---")
    print(f"Query: '{query}'")

    # Step 1: Vector retrieval
    vector_docs = vectorstore.similarity_search(query, k=5)
    print(f"\nVector search top 3:")
    for i, doc in enumerate(vector_docs[:3]):
        print(f"  {i+1}. {doc.page_content[:80]}...")

    # Step 2: BM25 retrieval
    tokenized_corpus = [doc.page_content.lower().split() for doc in
                        [Document(page_content=text) for text in KNOWLEDGE_BASE]]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top 5 by BM25 score
    top_indices = np.argsort(bm25_scores)[::-1][:5]
    bm25_docs = [Document(page_content=KNOWLEDGE_BASE[i]) for i in top_indices]

    print(f"\nBM25 search top 3:")
    for i, doc in enumerate(bm25_docs[:3]):
        print(f"  {i+1}. {doc.page_content[:80]}...")

    # Step 3: Reciprocal Rank Fusion (RRF)
    # Combine rankings from both methods
    def reciprocal_rank_fusion(doc_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Combine multiple ranked lists using RRF algorithm"""
        doc_scores = {}

        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                content = doc.page_content
                if content not in doc_scores:
                    doc_scores[content] = 0
                doc_scores[content] += 1 / (k + rank + 1)

        # Sort by score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [Document(page_content=content) for content, score in ranked]

    fused_docs = reciprocal_rank_fusion([vector_docs, bm25_docs])

    print(f"\nFused results (top 3 after RRF):")
    for i, doc in enumerate(fused_docs[:3]):
        print(f"  {i+1}. {doc.page_content[:80]}...")

    # Step 4: Generate answer
    context = "\n\n".join([doc.page_content for doc in fused_docs[:5]])
    prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.invoke(prompt).content

    print(f"\nAnswer: {answer}")
    return answer


# --- 12E: Reranking with Cross-Encoder ---
def reranking_rag(query: str) -> str:
    """
    Reranking: Retrieve top-k candidates with embeddings, then rerank with a cross-encoder.

    WHY: Embedding models score query-doc independently.
         Cross-encoder sees both query+doc together → more accurate relevance.

    NOTE: For true cross-encoder reranking, you'd use a model like:
          sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2

    Here we simulate reranking by using LLM to score relevance.
    """
    print("\n--- 12E: Reranking (LLM-based simulation) ---")
    print(f"Query: '{query}'")

    # Step 1: Initial retrieval (cast wide net)
    initial_docs = vectorstore.similarity_search(query, k=10)

    print(f"\nInitial retrieval: {len(initial_docs)} chunks")

    # Step 2: Rerank using LLM to score relevance
    rerank_prompt_template = """On a scale of 1-10, how relevant is this passage to answering the question?
Output only the number.

Question: {query}

Passage: {passage}

Relevance (1-10):"""

    scored_docs = []
    for doc in initial_docs:
        prompt = rerank_prompt_template.format(query=query, passage=doc.page_content)
        try:
            score_text = llm.invoke(prompt).content.strip()
            # Extract first number
            score = float(''.join(filter(str.isdigit, score_text.split()[0])))
            scored_docs.append((doc, score))
        except:
            scored_docs.append((doc, 5.0))  # Default score if parsing fails

    # Sort by score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    print(f"\nReranked results (top 3 with scores):")
    for i, (doc, score) in enumerate(scored_docs[:3]):
        print(f"  {i+1}. [Score: {score}/10] {doc.page_content[:80]}...")

    # Step 3: Use top-k after reranking
    top_docs = [doc for doc, score in scored_docs[:3]]
    context = "\n\n".join([doc.page_content for doc in top_docs])
    prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.invoke(prompt).content

    print(f"\nAnswer: {answer}")
    return answer


# --- 12F: Self-RAG (Critique and Refine) ---
def self_rag(query: str) -> str:
    """
    Self-RAG: Generate answer, critique it, retrieve more context if needed, refine.

    WHY: Catches hallucinations and missing information.
    """
    print("\n--- 12F: Self-RAG (Critique and Refine) ---")
    print(f"Query: '{query}'")

    # Step 1: Initial retrieval + answer
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    answer_prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    initial_answer = llm.invoke(answer_prompt).content

    print(f"\nInitial answer: {initial_answer}")

    # Step 2: Critique the answer
    critique_prompt = f"""Critique this answer. Is it:
1. Factually supported by the context?
2. Complete (answers all parts of the question)?
3. Specific (includes concrete details, not vague)?

Question: {query}
Answer: {initial_answer}
Context: {context}

Critique (identify any issues):"""

    critique = llm.invoke(critique_prompt).content
    print(f"\nCritique: {critique}")

    # Step 3: Check if refinement is needed
    needs_refinement = any(word in critique.lower() for word in ['missing', 'incomplete', 'vague', 'not supported', 'lacks'])

    if needs_refinement:
        print("\n→ Critique indicates issues. Retrieving additional context...")

        # Retrieve more docs
        additional_docs = vectorstore.similarity_search(query, k=5)
        expanded_context = "\n\n".join([doc.page_content for doc in additional_docs])

        # Refine answer
        refine_prompt = f"""The initial answer had issues: {critique}

Using this expanded context, provide a refined, complete answer.

Context:\n{expanded_context}\n\nQuestion: {query}\n\nRefined Answer:"""

        refined_answer = llm.invoke(refine_prompt).content
        print(f"\nRefined answer: {refined_answer}")
        return refined_answer
    else:
        print("\n→ Critique is positive. No refinement needed.")
        return initial_answer


# --- 12G: Step-Back Prompting ---
def step_back_rag(query: str) -> str:
    """
    Step-Back: Generate a broader, more general question, retrieve for it, then answer original.

    WHY: Specific questions might miss relevant high-level context.

    Example:
      Original: "What's the token expiration time?"
      Step-back: "How does authentication work in the system?"
      → Retrieves broader auth context, then answers specific question.
    """
    print("\n--- 12G: Step-Back Prompting ---")
    print(f"Original query: '{query}'")

    # Step 1: Generate step-back question
    step_back_prompt = f"""Given this specific question, generate a broader, more general question
that would help provide context for answering it.

Specific question: {query}

Broader question:"""

    broader_question = llm.invoke(step_back_prompt).content.strip()
    print(f"\nStep-back (broader) question: '{broader_question}'")

    # Step 2: Retrieve using broader question
    broad_docs = vectorstore.similarity_search(broader_question, k=3)

    # Also retrieve for original question
    specific_docs = vectorstore.similarity_search(query, k=2)

    # Combine (prioritize specific)
    all_docs = specific_docs + [doc for doc in broad_docs if doc.page_content not in [d.page_content for d in specific_docs]]

    print(f"\nRetrieved {len(all_docs)} chunks (specific + broad context)")

    # Step 3: Answer original question using combined context
    context = "\n\n".join([doc.page_content for doc in all_docs[:5]])
    answer_prompt = f"""Using this context (which includes both specific and broader background information),
answer the specific question.

Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

    answer = llm.invoke(answer_prompt).content
    print(f"\nAnswer: {answer}")
    return answer


# --- Comparison Demo ---
def compare_all_strategies(query: str):
    """Run all strategies side-by-side for comparison"""
    print("\n" + "="*70)
    print(f"COMPARISON: All Strategies for Query: '{query}'")
    print("="*70)

    strategies = [
        ("Baseline RAG", baseline_rag),
        ("HyDE", hyde_rag),
        ("Multi-Query", multi_query_rag),
        ("Fusion (Vector+BM25)", fusion_retrieval),
        ("Reranking", reranking_rag),
        ("Self-RAG", self_rag),
        ("Step-Back", step_back_rag),
    ]

    results = {}
    for name, strategy_fn in strategies:
        print("\n" + "-"*70)
        try:
            results[name] = strategy_fn(query)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = f"Error: {e}"

    print("\n" + "="*70)
    print("SUMMARY OF ANSWERS:")
    print("="*70)
    for name, answer in results.items():
        print(f"\n{name}:")
        print(f"  {answer[:150]}...")


def main():
    print("=" * 70)
    print("Feature 12: Advanced RAG Strategies")
    print("=" * 70)

    setup()

    # Test query that benefits from advanced techniques
    test_query = "How does the system authenticate users?"

    # Run individual strategies
    baseline_rag(test_query)
    hyde_rag(test_query)
    multi_query_rag(test_query)
    fusion_retrieval(test_query)
    reranking_rag(test_query)
    self_rag(test_query)
    step_back_rag(test_query)

    # Full comparison
    print("\n\n")
    compare_all_strategies("What happens during high traffic?")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  1. HyDE: Generate hypothetical answer → better embedding match")
    print("  2. Multi-Query: Multiple query variations → higher recall")
    print("  3. Fusion: Vector + BM25 → semantic + keyword matching")
    print("  4. Reranking: Cross-encoder scores query+doc together → precision")
    print("  5. Self-RAG: Critique answer → catch hallucinations")
    print("  6. Step-Back: Broader question → more context")
    print("  7. Combine strategies for production (e.g., Fusion + Reranking)")
    print("=" * 70)


if __name__ == "__main__":
    main()
