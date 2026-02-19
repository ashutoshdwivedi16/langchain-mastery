"""
Feature 11: Semantic Chunking Strategies for Long Documents

Demonstrates advanced chunking techniques for processing large documents (100+ pages)
while maintaining context and semantic coherence.

Covers:
- Naive token-based chunking (baseline — NOT recommended for production)
- Recursive character splitting (respects code/markdown structure)
- Semantic chunking via embeddings (groups related content)
- Layout-aware chunking (preserves document structure)
- Overlapping context windows
- Global summary + chunk strategy
- Embedding-based chunk clustering

Prerequisites:
    pip install langchain langchain-community langchain-text-splitters
    pip install langchain-huggingface sentence-transformers
    pip install scikit-learn  # for clustering
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import hashlib


# --- Sample Long Document (simulates 100+ page technical document) ---
SAMPLE_DOCUMENT = """
# Executive Summary

This document outlines the architecture of a distributed payment processing system.
The system handles millions of transactions daily across multiple geographic regions.

Key findings: 99.99% uptime achieved, average latency under 200ms, compliance with PCI-DSS standards.

# Chapter 1: System Architecture

## 1.1 Overview

The payment system consists of three primary layers:
1. API Gateway Layer - handles authentication and rate limiting
2. Processing Layer - executes payment transactions
3. Data Layer - ensures ACID compliance for financial records

The API Gateway uses JWT tokens for authentication and implements sliding window rate limiting
to prevent abuse. Maximum 1000 requests per minute per client.

## 1.2 API Gateway Design

The gateway is built on NGINX with custom Lua modules. It performs:
- Request validation
- Token verification
- Rate limiting via Redis
- Request routing to backend services

Configuration example:
```nginx
location /api/payment {
    access_by_lua_block {
        local jwt = require "resty.jwt"
        local token = ngx.var.http_authorization
        -- validation logic here
    }
    proxy_pass http://payment_backend;
}
```

Security considerations: All API endpoints require TLS 1.3. Certificate rotation happens
automatically every 90 days via Let's Encrypt integration.

## 1.3 Processing Layer Architecture

The processing layer uses event sourcing to maintain transaction history. Every state change
is recorded as an immutable event in the event store.

Benefits:
- Complete audit trail for compliance
- Ability to replay events for debugging
- Temporal queries (state at any point in time)

Drawbacks:
- Higher storage requirements
- Complexity in event schema evolution
- Need for snapshot mechanism for performance

### 1.3.1 Event Store Implementation

We use Apache Kafka as the event store backbone. Each payment goes through these stages:
1. INITIATED - customer submits payment
2. VALIDATED - fraud checks pass
3. AUTHORIZED - bank pre-authorization succeeds
4. CAPTURED - funds transferred
5. SETTLED - batch settlement with acquiring bank

Kafka topic configuration: 30 partitions, replication factor 3, min.insync.replicas=2

# Chapter 2: Data Management

## 2.1 Database Strategy

We use a polyglot persistence approach:
- PostgreSQL for transactional data (payments, accounts)
- Cassandra for time-series analytics (transaction logs)
- Redis for session and cache management
- Elasticsearch for full-text search

PostgreSQL configuration optimizations:
```sql
-- Partition tables by date for efficient archival
CREATE TABLE payments (
    id UUID PRIMARY KEY,
    amount DECIMAL(19,4),
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

CREATE INDEX idx_payment_status ON payments(status) WHERE status != 'SETTLED';
```

## 2.2 Backup and Recovery

Daily full backups to S3 with 7-year retention for compliance.
Point-in-time recovery window: 30 days.

Recovery Time Objective (RTO): 4 hours
Recovery Point Objective (RPO): 15 minutes

# Chapter 3: Security and Compliance

## 3.1 PCI-DSS Compliance

Our system is PCI-DSS Level 1 compliant. Key requirements:
- Cardholder data is tokenized immediately at point of entry
- No plaintext card numbers stored anywhere in the system
- All logs are sanitized to remove sensitive data
- Annual security audits by qualified assessors

Tokenization flow:
1. Frontend sends card data directly to tokenization service (bypasses our backend)
2. Tokenization service returns a one-time token
3. Backend uses token for payment processing
4. Token expires after single use

## 3.2 Fraud Detection

Machine learning model analyzes transactions in real-time:
- Features: transaction amount, merchant category, user location, device fingerprint, time of day
- Model: XGBoost classifier, retrained weekly on latest fraud patterns
- False positive rate: <2%
- Detection rate: >95%

High-risk transactions trigger step-up authentication (SMS or biometric).

# Chapter 4: Performance Optimization

## 4.1 Caching Strategy

Three-tier caching:
1. L1: Application-level cache (Caffeine, 5-minute TTL)
2. L2: Redis cluster (1-hour TTL)
3. L3: CDN edge cache (static content only)

Cache invalidation uses event-driven updates via Kafka. When a payment status changes,
an event triggers cache invalidation across all nodes.

## 4.2 Database Query Optimization

Most expensive query before optimization:
```sql
SELECT p.*, u.email, m.name
FROM payments p
JOIN users u ON p.user_id = u.id
JOIN merchants m ON p.merchant_id = m.id
WHERE p.created_at >= NOW() - INTERVAL '30 days'
  AND p.status = 'SETTLED'
ORDER BY p.amount DESC
LIMIT 100;
```

Execution time: 4.2 seconds (full table scan)

After optimization (materialized view + index):
Execution time: 23ms (index-only scan)

# Appendix A: Deployment Architecture

Infrastructure: Kubernetes on AWS EKS
Regions: us-east-1 (primary), eu-west-1 (secondary), ap-southeast-1 (tertiary)

Auto-scaling configuration:
- Min pods: 10 per service
- Max pods: 200 per service
- Target CPU: 70%
- Scale-up: add 50% pods every 30 seconds
- Scale-down: remove 10% pods every 2 minutes

# Appendix B: Monitoring and Observability

Metrics: Prometheus + Grafana
Logs: ELK stack (Elasticsearch, Logstash, Kibana)
Traces: Jaeger for distributed tracing
Alerts: PagerDuty integration

SLOs:
- API availability: 99.99%
- P95 latency: <200ms
- Error rate: <0.1%

# Appendix C: Incident Response Runbook

Severity 1 (Complete Outage):
1. Page on-call engineer immediately
2. Start war room (Zoom link in wiki)
3. Update status page
4. Activate incident commander
5. Begin triage within 5 minutes

Severity 2 (Degraded Service):
1. Create incident ticket
2. Notify team lead
3. Investigate within 30 minutes
4. Provide update every hour

Post-incident review required for all Sev1/Sev2 incidents.
"""


# --- 11A: Naive Token-Based Chunking (BASELINE — NOT RECOMMENDED) ---
def naive_chunking(text: str, chunk_size: int = 500) -> List[Document]:
    """
    Simple character-based splitting. PROBLEM: Breaks mid-sentence, mid-code block, etc.
    This is the WRONG approach for production but shown as a baseline.
    """
    print("\n--- 11A: Naive Character Chunking (Baseline) ---")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,  # 10% overlap to preserve some context
        separator=" ",
        length_function=len,
    )

    chunks = splitter.create_documents([text])

    print(f"  Chunks created: {len(chunks)}")
    print(f"  Chunk 0 preview: {chunks[0].page_content[:100]}...")
    print(f"  Chunk 1 preview: {chunks[1].page_content[:100]}...")

    # PROBLEM: Check if code blocks are broken
    for i, chunk in enumerate(chunks):
        if "```" in chunk.page_content:
            open_count = chunk.page_content.count("```")
            if open_count % 2 != 0:
                print(f"  WARNING: Chunk {i} has BROKEN code block (odd number of ```)")
                break

    return chunks


# --- 11B: Recursive Character Splitting (RESPECTS STRUCTURE) ---
def recursive_chunking(text: str, chunk_size: int = 1000) -> List[Document]:
    """
    Recursive splitting that respects document structure (paragraphs, code blocks, lists).
    This is the RECOMMENDED baseline for most text documents.
    """
    print("\n--- 11B: Recursive Character Splitting (Respects Structure) ---")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,  # 20% overlap for context preservation
        separators=[
            "\n\n\n",  # Multiple blank lines (chapter breaks)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence breaks
            " ",       # Word breaks
            "",        # Character breaks (last resort)
        ],
        length_function=len,
    )

    chunks = splitter.create_documents([text])

    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # Check that code blocks are preserved
    broken_blocks = 0
    for i, chunk in enumerate(chunks):
        open_count = chunk.page_content.count("```")
        if open_count % 2 != 0:
            broken_blocks += 1

    print(f"  Broken code blocks: {broken_blocks} (should be 0 or minimal)")

    return chunks


# --- 11C: Layout-Aware Chunking (MARKDOWN HEADERS) ---
def layout_aware_chunking(text: str) -> List[Document]:
    """
    Split by document structure (headings). Each chunk corresponds to a logical section.
    This respects the document's semantic organization.
    """
    print("\n--- 11C: Layout-Aware Chunking (Markdown Headers) ---")

    # Split by headers first
    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    header_chunks = markdown_splitter.split_text(text)

    print(f"  Header-based chunks: {len(header_chunks)}")

    # Now split large sections further if needed
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    final_chunks = []
    for chunk in header_chunks:
        if len(chunk.page_content) > 1000:
            # Split large sections
            sub_chunks = recursive_splitter.split_documents([chunk])
            # Preserve header metadata
            for sub in sub_chunks:
                sub.metadata.update(chunk.metadata)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    print(f"  Final chunks after recursive split: {len(final_chunks)}")

    # Show metadata structure
    if final_chunks:
        print(f"  Sample metadata: {final_chunks[5].metadata}")
        print(f"  Sample content: {final_chunks[5].page_content[:150]}...")

    return final_chunks


# --- 11D: Semantic Chunking via Embeddings ---
def semantic_chunking(text: str, similarity_threshold: float = 0.5) -> List[Document]:
    """
    Split text into sentences, embed them, then group semantically similar sentences.
    This creates chunks based on meaning rather than structure.
    """
    print("\n--- 11D: Semantic Chunking (Embedding-Based) ---")

    # Step 1: Split into sentences
    sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 20]
    print(f"  Total sentences: {len(sentences)}")

    # Step 2: Embed all sentences
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    sentence_embeddings = embeddings_model.embed_documents(sentences)
    embeddings_array = np.array(sentence_embeddings)

    # Step 3: Compute cosine similarity between consecutive sentences
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Step 4: Group sentences where similarity drops below threshold
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings_array[i-1], embeddings_array[i])

        if similarity >= similarity_threshold:
            # Continue current chunk
            current_chunk.append(sentences[i])
        else:
            # Start new chunk
            chunks.append(Document(
                page_content=". ".join(current_chunk) + ".",
                metadata={"chunk_method": "semantic", "sentences": len(current_chunk)}
            ))
            current_chunk = [sentences[i]]

    # Add final chunk
    if current_chunk:
        chunks.append(Document(
            page_content=". ".join(current_chunk) + ".",
            metadata={"chunk_method": "semantic", "sentences": len(current_chunk)}
        ))

    print(f"  Semantic chunks created: {len(chunks)}")
    print(f"  Avg sentences per chunk: {sum(c.metadata['sentences'] for c in chunks) // len(chunks)}")

    return chunks


# --- 11E: Overlapping Context Windows ---
def overlapping_context_windows(
    chunks: List[Document],
    overlap_size: int = 200
) -> List[Document]:
    """
    Add overlapping context from adjacent chunks to preserve continuity.
    Each chunk gets a prefix from the previous chunk and a suffix from the next chunk.
    """
    print("\n--- 11E: Overlapping Context Windows ---")

    windowed_chunks = []

    for i, chunk in enumerate(chunks):
        # Get overlap from previous chunk (suffix)
        prev_overlap = ""
        if i > 0:
            prev_text = chunks[i-1].page_content
            prev_overlap = prev_text[-overlap_size:] if len(prev_text) > overlap_size else prev_text

        # Get overlap from next chunk (prefix)
        next_overlap = ""
        if i < len(chunks) - 1:
            next_text = chunks[i+1].page_content
            next_overlap = next_text[:overlap_size] if len(next_text) > overlap_size else next_text

        # Create windowed chunk
        windowed_content = f"[PREV CONTEXT: ...{prev_overlap}]\n\n{chunk.page_content}\n\n[NEXT CONTEXT: {next_overlap}...]"

        windowed_chunks.append(Document(
            page_content=windowed_content,
            metadata={
                **chunk.metadata,
                "has_prev_context": i > 0,
                "has_next_context": i < len(chunks) - 1,
            }
        ))

    print(f"  Windowed chunks created: {len(windowed_chunks)}")
    print(f"  Sample windowed chunk:\n{windowed_chunks[3].page_content[:300]}...")

    return windowed_chunks


# --- 11F: Global Summary + Chunk Strategy ---
def summarize_document(text: str) -> str:
    """
    Create a global summary to provide document-wide context.
    In production, use an LLM (GPT-4, Claude) for summarization.
    Here we use a simple extractive summary (first + last paragraphs).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

    # Extract key sections
    summary_parts = []

    # Executive summary (if exists)
    if "Executive Summary" in text or "# Executive Summary" in text:
        exec_summary = [p for p in paragraphs if "Executive Summary" in p or "Key findings" in p]
        summary_parts.extend(exec_summary[:2])

    # First substantive paragraph
    summary_parts.append(paragraphs[1] if len(paragraphs) > 1 else paragraphs[0])

    # Last paragraph (often contains conclusions)
    summary_parts.append(paragraphs[-1])

    summary = "\n\n".join(summary_parts)
    return f"DOCUMENT SUMMARY:\n{summary}\n\n---\n"


def chunks_with_global_summary(chunks: List[Document], summary: str) -> List[Document]:
    """
    Prepend global summary to each chunk so the LLM has document-wide context.
    """
    print("\n--- 11F: Global Summary + Chunks ---")

    summarized_chunks = []
    for chunk in chunks:
        summarized_chunks.append(Document(
            page_content=f"{summary}\nCHUNK CONTENT:\n{chunk.page_content}",
            metadata={**chunk.metadata, "has_global_summary": True}
        ))

    print(f"  Chunks with summary: {len(summarized_chunks)}")
    print(f"  Summary preview:\n{summary[:200]}...")

    return summarized_chunks


# --- 11G: Embedding-Based Chunk Clustering ---
def cluster_chunks(chunks: List[Document], n_clusters: int = 5) -> Dict[int, List[Document]]:
    """
    Cluster chunks by semantic similarity using KMeans on embeddings.
    Useful for grouping related sections across a large document.
    """
    print("\n--- 11G: Embedding-Based Chunk Clustering ---")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Embed all chunks
    chunk_texts = [c.page_content for c in chunks]
    embeddings = embeddings_model.embed_documents(chunk_texts)
    embeddings_array = np.array(embeddings)

    # Cluster
    kmeans = KMeans(n_clusters=min(n_clusters, len(chunks)), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)

    # Group chunks by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[i])

    print(f"  Clusters created: {len(clusters)}")
    for cluster_id, cluster_chunks in clusters.items():
        print(f"  Cluster {cluster_id}: {len(cluster_chunks)} chunks")

    # Show representative chunk from each cluster
    print("\n  Representative chunks from each cluster:")
    for cluster_id, cluster_chunks in clusters.items():
        rep_chunk = cluster_chunks[0].page_content[:100]
        print(f"  Cluster {cluster_id}: {rep_chunk}...")

    return clusters


# --- 11H: RAG Query with Chunking Strategy Comparison ---
def rag_query_comparison(text: str, query: str):
    """
    Compare retrieval accuracy across different chunking strategies.
    """
    print("\n--- 11H: RAG Query with Different Chunking Strategies ---")
    print(f"  Query: '{query}'")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    strategies = {
        "Naive": naive_chunking(text, chunk_size=500),
        "Recursive": recursive_chunking(text, chunk_size=1000),
        "Layout-Aware": layout_aware_chunking(text),
    }

    print("\n  Retrieval Results (Top 2 chunks per strategy):\n")

    for strategy_name, chunks in strategies.items():
        # Build vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Query
        results = vectorstore.similarity_search(query, k=2)

        print(f"  --- {strategy_name} Strategy ---")
        for i, doc in enumerate(results):
            print(f"    Result {i+1}: {doc.page_content[:150]}...")
            if doc.metadata:
                print(f"    Metadata: {doc.metadata}")
        print()


def main():
    print("=" * 70)
    print("Feature 11: Semantic Chunking Strategies for Long Documents")
    print("=" * 70)

    # 11A: Naive chunking
    naive_chunks = naive_chunking(SAMPLE_DOCUMENT, chunk_size=500)

    # 11B: Recursive chunking
    recursive_chunks = recursive_chunking(SAMPLE_DOCUMENT, chunk_size=1000)

    # 11C: Layout-aware chunking
    layout_chunks = layout_aware_chunking(SAMPLE_DOCUMENT)

    # 11D: Semantic chunking
    semantic_chunks = semantic_chunking(SAMPLE_DOCUMENT, similarity_threshold=0.5)

    # 11E: Overlapping context windows
    windowed_chunks = overlapping_context_windows(recursive_chunks[:10], overlap_size=150)

    # 11F: Global summary + chunks
    summary = summarize_document(SAMPLE_DOCUMENT)
    summary_chunks = chunks_with_global_summary(recursive_chunks[:5], summary)

    # 11G: Clustering
    clusters = cluster_chunks(recursive_chunks, n_clusters=5)

    # 11H: RAG query comparison
    rag_query_comparison(
        SAMPLE_DOCUMENT,
        "How does the system handle fraud detection?"
    )

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  1. Naive chunking breaks structure — avoid for production")
    print("  2. RecursiveCharacterTextSplitter is the baseline for most documents")
    print("  3. Layout-aware chunking (headers) preserves semantic sections")
    print("  4. Semantic chunking groups by meaning, not structure")
    print("  5. Overlapping windows preserve continuity across chunks")
    print("  6. Global summary provides document-wide context to each chunk")
    print("  7. Clustering identifies related chunks across the document")
    print("=" * 70)


if __name__ == "__main__":
    main()
