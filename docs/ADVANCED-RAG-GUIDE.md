# Advanced RAG Strategies — Production Guide

## The RAG Accuracy Problem

Basic RAG (embed docs → retrieve top-k → generate answer) achieves **60-70% accuracy** on most datasets. Why?

### Common Failure Modes

1. **Query-Document Mismatch**
   ```
   User query: "How does auth work?"
   Document:    "JWT tokens with 1-hour expiration..."

   Problem: Query has no word overlap with doc → poor embedding match
   ```

2. **Retrieval Noise**
   ```
   Top 3 results:
   1. [Relevant] JWT authentication details
   2. [Irrelevant] Rate limiting implementation  ← NOISE
   3. [Relevant] Token refresh flow
   ```

3. **Missing Context**
   ```
   Chunk 1: "Tokens expire after 1 hour"
   Chunk 2: "Refresh tokens last 30 days"

   Retrieved: Only Chunk 1
   Answer: "Tokens expire after 1 hour" ← INCOMPLETE (missing refresh info)
   ```

4. **Hallucination**
   ```
   Retrieved: "Tokens expire after 1 hour"
   LLM answer: "Tokens expire after 2 hours and can be extended indefinitely"
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                NOT IN CONTEXT — HALLUCINATED
   ```

**Advanced RAG strategies push accuracy to 85-95%** by addressing these failure modes.

---

## Strategy 1: HyDE (Hypothetical Document Embeddings)

### The Problem
User queries are short and vague. Documents are detailed and technical.

```
Query embedding:    [0.2, 0.5, ..., 0.3]  ← "How does auth work?" (4 words)
Document embedding: [0.8, 0.1, ..., 0.7]  ← "JWT tokens with asymmetric..."
Cosine similarity: 0.42  ← POOR MATCH
```

### The Solution
Generate a **hypothetical answer** first, then embed that for retrieval.

```python
# Step 1: Generate hypothetical answer (without seeing docs)
hypothetical = llm.invoke("""
Write a detailed technical answer to: How does auth work?
Use specific terms like JWT, tokens, expiration, refresh.
""")

# hypothetical: "Authentication uses JWT tokens with 1-hour expiration.
#                Refresh tokens allow extending sessions to 30 days..."

# Step 2: Embed the hypothetical answer
embedding = embeddings.embed_query(hypothetical)

# Step 3: Retrieve using this embedding
docs = vectorstore.similarity_search_by_vector(embedding, k=3)

# Step 4: Generate final answer from retrieved docs
answer = llm.invoke(f"Context: {docs}\n\nQuestion: {query}")
```

### Why It Works
The hypothetical answer uses **technical terminology** that matches how the actual docs are written.

```
Hypothetical embedding: [0.79, 0.12, ..., 0.68]
Document embedding:     [0.80, 0.11, ..., 0.70]
Cosine similarity: 0.94  ← EXCELLENT MATCH
```

### When to Use
- User queries are natural language ("How do I...?", "What is...?")
- Documents are technical (API docs, architecture specs, code)
- Domain has specific jargon (medical, legal, engineering)

### Cost
2x LLM calls (hypothetical + final answer) → **2x cost vs. baseline**

---

## Strategy 2: Multi-Query Retrieval

### The Problem
Single query might miss relevant docs due to wording.

```
Query: "How does the system scale?"

Misses docs about:
- "High availability architecture"
- "Load balancing strategy"
- "Multi-region deployment"

Why? No lexical overlap with "scale"
```

### The Solution
Generate multiple query variations, retrieve for each, combine results.

```python
# Step 1: Generate variations
variations = llm.invoke("""
Generate 3 alternative phrasings of: How does the system scale?

1. What is the scalability approach?
2. How does the architecture handle increased load?
3. What mechanisms support high traffic volumes?
""")

# Step 2: Retrieve for each
all_docs = []
for query in variations:
    docs = vectorstore.similarity_search(query, k=3)
    all_docs.extend(docs)

# Step 3: Deduplicate
unique_docs = list(set(all_docs))  # Remove duplicates

# Step 4: Generate answer from combined results
answer = llm.invoke(f"Context: {unique_docs}\n\nQuestion: {original_query}")
```

### Why It Works
Different phrasings retrieve different docs → **higher recall**.

```
"How does the system scale?" → Doc 1, Doc 3, Doc 5
"Scalability approach"       → Doc 2, Doc 3, Doc 7
"Handle increased load"      → Doc 1, Doc 4, Doc 6

Combined: Doc 1, 2, 3, 4, 5, 6, 7  ← 7 unique docs from 3 queries
```

### When to Use
- Queries can be phrased many ways (ambiguous or broad questions)
- Missing relevant docs is worse than retrieving too many
- You have budget for 3-5x retrieval calls

### Cost
- 1 LLM call (generate variations)
- 3-5x retrieval calls
- Total: **~2x cost vs. baseline**

---

## Strategy 3: Fusion Retrieval (Vector + BM25)

### The Problem
**Vector search** is great for semantic similarity but misses exact keyword matches.

```
Query: "PCI-DSS compliance"

Vector search finds:
- "Security audits and assessments"  ← Semantically related
- "Compliance requirements overview" ← Semantically related

Misses:
- "PCI-DSS Level 1 compliance is maintained..." ← Exact keyword match!
```

### The Solution
Combine **vector similarity** with **BM25 keyword search**.

```python
# Step 1: Vector retrieval
vector_docs = vectorstore.similarity_search(query, k=10)

# Step 2: BM25 retrieval
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(query_tokens)
bm25_docs = top_k_by_bm25(bm25_scores, k=10)

# Step 3: Reciprocal Rank Fusion (RRF)
def rrf_score(rank, k=60):
    return 1 / (k + rank)

doc_scores = {}
for rank, doc in enumerate(vector_docs):
    doc_scores[doc] = doc_scores.get(doc, 0) + rrf_score(rank)

for rank, doc in enumerate(bm25_docs):
    doc_scores[doc] = doc_scores.get(doc, 0) + rrf_score(rank)

# Step 4: Sort by combined score
fused_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

# Step 5: Use top-k for generation
answer = llm.invoke(f"Context: {fused_docs[:5]}\n\nQuestion: {query}")
```

### Why It Works
Combines two complementary signals:
- **Vector**: Semantic similarity (handles synonyms, paraphrasing)
- **BM25**: Exact term matching (handles technical terms, acronyms)

### When to Use
- Queries contain specific technical terms or acronyms
- Documents have important keywords that must be matched
- You have compute budget for BM25 (O(n) per query)

### Cost
- No extra LLM calls
- 2x retrieval (vector + BM25)
- **Same cost as baseline** (LLM-wise), but more compute

---

## Strategy 4: Reranking with Cross-Encoder

### The Problem
Embedding models score **query** and **document** independently.

```
Embedding model:
  query_emb = encode(query)           ← Independent
  doc_emb   = encode(doc)             ← Independent
  similarity = cosine(query_emb, doc_emb)
```

This misses **interaction features** between query and doc.

### The Solution
Use a **cross-encoder** that sees both together.

```python
# Step 1: Initial retrieval (cast wide net)
candidate_docs = vectorstore.similarity_search(query, k=20)

# Step 2: Rerank with cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc.page_content] for doc in candidate_docs]
scores = reranker.predict(pairs)

# Step 3: Sort by cross-encoder score
ranked_docs = [doc for _, doc in sorted(zip(scores, candidate_docs), reverse=True)]

# Step 4: Use top-k after reranking
answer = llm.invoke(f"Context: {ranked_docs[:3]}\n\nQuestion: {query}")
```

### Why It Works
Cross-encoder sees **query + document together** → more accurate relevance.

```
Embedding model:
  "auth" matches "JWT" → score = 0.6

Cross-encoder:
  Input: "[CLS] How does auth work? [SEP] JWT tokens with 1-hour expiration... [SEP]"
  Attention across both → score = 0.94
```

### When to Use
- **Precision matters more than recall** (e.g., legal QA, medical QA)
- You can afford the latency (cross-encoder is 10-100x slower than embeddings)
- Top-k from initial retrieval must include the answer (high initial recall)

### Cost
- 20x embedding retrievals (instead of 3-5)
- Cross-encoder forward pass for each pair
- **~5-10x slower** than baseline, but **higher precision**

---

## Strategy 5: Self-RAG (Critique and Refine)

### The Problem
LLMs hallucinate — they make up facts not in the retrieved context.

```
Retrieved: "Tokens expire after 1 hour"
Answer:    "Tokens expire after 2 hours and can be refreshed indefinitely"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            NOT SUPPORTED BY CONTEXT
```

### The Solution
Generate answer, **critique** it, retrieve more if needed, **refine**.

```python
# Step 1: Initial RAG
docs = vectorstore.similarity_search(query, k=3)
answer = llm.invoke(f"Context: {docs}\n\nQuestion: {query}")

# Step 2: Critique
critique = llm.invoke(f"""
Critique this answer:
1. Is it factually supported by the context?
2. Is it complete?
3. Is it specific?

Context: {docs}
Question: {query}
Answer: {answer}

Critique:
""")

# Step 3: Check if refinement needed
if "not supported" in critique or "incomplete" in critique:
    # Retrieve more context
    more_docs = vectorstore.similarity_search(query, k=10)

    # Refine answer
    refined = llm.invoke(f"""
    The initial answer had issues: {critique}

    Context: {more_docs}
    Question: {query}

    Provide a complete, factually accurate answer:
    """)
    return refined
else:
    return answer
```

### Why It Works
**Catches hallucinations and missing information** before returning to user.

### When to Use
- **Factual accuracy is critical** (legal, medical, financial QA)
- Cost of hallucination is high (customer-facing, compliance)
- You can afford 2-3x LLM calls

### Cost
- 2-3x LLM calls (initial + critique + optional refine)
- **2-3x cost vs. baseline**

---

## Strategy 6: Step-Back Prompting

### The Problem
Specific questions miss broader context.

```
Query: "What's the token expiration time?"

Retrieved: "Tokens expire after 1 hour"

Missing context:
- What are tokens used for?
- How do refresh tokens work?
- Why 1 hour (security rationale)?
```

### The Solution
Generate a **broader question**, retrieve for it, then answer the specific question.

```python
# Step 1: Generate step-back (broader) question
broader_q = llm.invoke(f"""
Given this specific question, generate a broader question that provides context.

Specific: {query}
Broader:
""")

# broader_q: "How does the authentication system work?"

# Step 2: Retrieve for broader question
broad_docs = vectorstore.similarity_search(broader_q, k=3)

# Step 3: Also retrieve for specific question
specific_docs = vectorstore.similarity_search(query, k=2)

# Step 4: Combine both
all_docs = specific_docs + broad_docs

# Step 5: Answer specific question using combined context
answer = llm.invoke(f"""
Context (specific + background):
{all_docs}

Question: {query}

Answer:
""")
```

### Why It Works
Retrieves **high-level context** that helps understand the specific answer.

### When to Use
- Questions are about implementation details (need architectural context)
- Answers require background knowledge to understand
- Documents have hierarchical structure (overview → details)

### Cost
- 2x LLM calls (generate broader question + final answer)
- 2x retrieval (specific + broad)
- **~2x cost vs. baseline**

---

## Comparison Matrix

| Strategy | Accuracy Gain | Latency | Cost | Best For |
|----------|---------------|---------|------|----------|
| **HyDE** | +10-15% | 2x | 2x LLM | Technical docs, jargon-heavy |
| **Multi-Query** | +8-12% | 1.5x | 2x LLM + 3x retrieval | Broad/ambiguous questions |
| **Fusion** | +5-10% | 1.3x | Same LLM, 2x retrieval | Keyword + semantic matching |
| **Reranking** | +15-20% | 5-10x | Same LLM, 20x retrieval | High precision required |
| **Self-RAG** | +12-18% | 2-3x | 2-3x LLM | Critical factual accuracy |
| **Step-Back** | +8-12% | 2x | 2x LLM, 2x retrieval | Detail questions needing context |

**Note:** Gains are approximate and depend on dataset, query types, and baseline quality.

---

## Production Recommendations

### Tier 1: Essential (Use on All Production RAG)

```python
# 1. Fusion Retrieval (Vector + BM25)
# 2. Basic reranking (LLM-based or lightweight cross-encoder)

vector_docs = vectorstore.similarity_search(query, k=10)
bm25_docs = bm25_search(query, k=10)
fused = reciprocal_rank_fusion([vector_docs, bm25_docs])

# Lightweight reranking (top 10 → top 3)
reranked = llm_rerank(query, fused[:10])
final_docs = reranked[:3]

answer = llm.invoke(f"Context: {final_docs}\n\nQuestion: {query}")
```

**Why:**
- Fusion adds minimal cost but catches keyword mismatches
- Basic reranking improves precision with acceptable latency

### Tier 2: High-Value Use Cases

```python
# Add HyDE for technical documentation
# Add Self-RAG for factual domains (legal, medical, financial)

if is_technical_domain:
    # HyDE
    hypothetical = generate_hypothetical_answer(query)
    docs = vectorstore.search_by_vector(embed(hypothetical), k=10)
else:
    docs = vectorstore.similarity_search(query, k=10)

# Fusion + Reranking
fused = fusion_retrieval(docs)
answer = llm.invoke(f"Context: {fused[:3]}\n\nQuestion: {query}")

if requires_factual_accuracy:
    # Self-RAG
    critique = llm.invoke(f"Critique this answer: {answer}")
    if needs_refinement(critique):
        answer = refine_with_more_context(query, answer, critique)

return answer
```

### Tier 3: Maximum Accuracy (Cost-Insensitive)

```python
# Combine multiple strategies
queries = multi_query_generation(query)                # 3-5 variations
broader_query = step_back_prompting(query)             # Broader context
all_queries = queries + [broader_query]

# Retrieve for all
all_docs = []
for q in all_queries:
    vector_docs = vectorstore.similarity_search(q, k=5)
    bm25_docs = bm25_search(q, k=5)
    all_docs.extend(vector_docs + bm25_docs)

# Fusion
fused = reciprocal_rank_fusion(all_docs)

# Heavy reranking
reranked = cross_encoder_rerank(query, fused[:50])  # Top 50 → Top 5

# Generate + Critique
answer = llm.invoke(f"Context: {reranked[:5]}\n\nQuestion: {query}")
critique = llm.invoke(f"Critique: {answer}")
final_answer = refine_if_needed(answer, critique)

return final_answer
```

**Cost:** ~10x baseline, but **90-95% accuracy**

---

## Evaluation

### Metrics

1. **Retrieval Precision@k**
   ```python
   # For each question, is the answer in top-k retrieved chunks?
   precision_at_3 = correct_retrievals / total_queries
   ```

2. **Answer Accuracy**
   ```python
   # Compare generated answer to ground truth
   # Can use LLM-as-judge or exact match
   accuracy = correct_answers / total_queries
   ```

3. **Faithfulness**
   ```python
   # Is the answer supported by the retrieved context?
   # Use NLI model or LLM-as-judge
   faithfulness = supported_answers / total_answers
   ```

4. **Latency**
   ```python
   # End-to-end time from query to answer
   p95_latency = np.percentile(latencies, 95)
   ```

### Benchmark Your Strategy

```python
test_set = [
    ("How does auth work?", "JWT tokens with 1-hour expiration..."),
    ("What is the rate limit?", "1000 requests per minute per API key"),
    # ... 20-30 more test cases
]

for strategy in [baseline, hyde, fusion, reranking]:
    correct = 0
    for query, ground_truth in test_set:
        answer = strategy(query)
        if is_correct(answer, ground_truth):
            correct += 1

    print(f"{strategy.__name__}: {correct}/{len(test_set)} = {100*correct/len(test_set):.1f}%")
```

---

## Common Pitfalls

### 1. Over-Engineering for Simple Queries

```python
# BAD: HyDE for "What is X?" when docs literally say "X is ..."
query = "What is JWT?"
docs = ["JWT is a JSON Web Token used for authentication..."]

# No need for HyDE — simple retrieval works fine
```

**Fix:** Use advanced strategies only when baseline fails.

### 2. Not Measuring Impact

```python
# BAD: Adding HyDE without measuring accuracy change
# You pay 2x cost but don't know if accuracy improved
```

**Fix:** A/B test on a held-out test set before deploying.

### 3. Ignoring Latency

```python
# BAD: Cross-encoder reranking on 100 candidates
# Takes 5-10 seconds per query → user-facing timeout
```

**Fix:** Use cross-encoder only for async workflows or with smaller candidate sets (10-20).

### 4. Combining Incompatible Strategies

```python
# BAD: HyDE + Multi-Query together
# Both generate text before retrieval → 5-10 LLM calls before seeing any docs
```

**Fix:** Choose **one** query transformation strategy per query.

---

## Decision Tree

```
START: Is baseline RAG accuracy <80%?
│
├─ YES: What's the main failure mode?
│  │
│  ├─ Query-document mismatch (user language ≠ doc language)
│  │  └─ Use HyDE
│  │
│  ├─ Missing relevant docs (low recall)
│  │  └─ Use Multi-Query or Fusion
│  │
│  ├─ Irrelevant docs in top-k (low precision)
│  │  └─ Use Reranking
│  │
│  ├─ Hallucination (making up facts)
│  │  └─ Use Self-RAG
│  │
│  └─ Specific questions need broader context
│     └─ Use Step-Back
│
└─ NO: Baseline is good enough
   └─ Add Fusion for robustness (minimal cost)
```

---

## Summary

| Problem | Strategy | How It Helps |
|---------|----------|--------------|
| Query-doc mismatch | HyDE | Embed hypothetical answer (matches doc language) |
| Low recall | Multi-Query | Multiple query variations retrieve more docs |
| Missing keywords | Fusion | BM25 catches exact term matches |
| Low precision | Reranking | Cross-encoder scores query+doc together |
| Hallucination | Self-RAG | Critique catches unsupported claims |
| Missing context | Step-Back | Broader question retrieves high-level context |

**Production stack:** Fusion + Reranking gets you 80-85% accuracy for ~2-3x cost.

Add HyDE or Self-RAG for specialized domains to reach 90%+.
