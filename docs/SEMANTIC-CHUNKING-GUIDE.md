# Semantic Chunking for Long Documents — Complete Guide

## The Problem: Context Window Limitations

Modern LLMs have context windows ranging from 4K tokens (older models) to 200K+ tokens (Claude 3.5, GPT-4 Turbo). However:

- **100-page document = ~50K-75K tokens** (assuming ~500-750 tokens/page)
- Even with large context windows, **retrieval precision degrades** as context increases
- **Cost scales linearly** with input tokens
- **Latency increases** with larger context
- **Needle-in-haystack problem**: LLMs struggle to find specific info in massive context

**Solution:** Chunk documents into smaller, semantically coherent pieces, embed them, retrieve only relevant chunks for queries.

---

## Strategy 1: Naive Character Splitting ❌

### What It Does
Splits text at character boundaries (e.g., every 500 characters) with optional overlap.

```python
CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator=" ")
```

### Problems

1. **Breaks mid-sentence:**
   ```
   Chunk 1: "The payment system consists of three layers: API Gateway, Processing L"
   Chunk 2: "ayer, and Data Layer. Each layer has distinct responsibilities..."
   ```

2. **Breaks code blocks:**
   ```python
   Chunk 1: "Configuration:\n```python\ndef process_payment(amount):\n    if amoun"
   Chunk 2: "t > 0:\n        return True\n```\n"
   ```
   → Chunk 1 has unclosed triple-backtick, breaking syntax highlighting and parsing

3. **Breaks tables:**
   ```
   Chunk 1: "| Metric | Value |\n|--------|----"
   Chunk 2: "---|\n| Latency | 200ms |\n"
   ```

### When to Use
**Never in production.** Only as a baseline to demonstrate what NOT to do.

---

## Strategy 2: Recursive Character Splitting ✅ (Baseline)

### What It Does
Recursively splits text using a hierarchy of separators, only falling back to smaller units when necessary.

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
)
```

**Splitting priority:**
1. Try to split at `\n\n\n` (chapter/section breaks)
2. If chunk still too large, split at `\n\n` (paragraph breaks)
3. If still too large, split at `\n` (line breaks)
4. If still too large, split at `. ` (sentence breaks)
5. Last resort: split at spaces or characters

### Why It Works

- **Respects document structure** — keeps paragraphs together
- **Preserves code blocks** — won't split inside triple-backticks
- **Maintains lists** — keeps bullet points together
- **Keeps sentences intact** — only splits at sentence boundaries if needed

### When to Use

**Default choice for most documents.** Works well for:
- Markdown/text files
- Code documentation
- Technical manuals
- Blog posts
- Research papers (plain text)

### Limitations

- No understanding of **semantic sections** (e.g., might split a 3-paragraph explanation across chunks)
- Doesn't preserve **document hierarchy** metadata (doesn't know "this chunk is from Section 2.3")

---

## Strategy 3: Layout-Aware Chunking ✅ (Structured Documents)

### What It Does
Splits by document structure (headings, sections, chapters) and preserves hierarchy as metadata.

```python
MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
    ]
)
```

**Result:**
```python
Document(
    page_content="Content of section 2.1...",
    metadata={
        "Chapter": "Chapter 2: Data Management",
        "Section": "2.1 Database Strategy"
    }
)
```

### Why It Works

- **Semantic boundaries** — each chunk corresponds to a logical section
- **Metadata preservation** — LLM can see which chapter/section a chunk came from
- **Hierarchical context** — "This answer comes from Section 2.1 of Chapter 2"
- **Table of Contents** — can generate ToC from metadata

### When to Use

Best for **well-structured documents** with clear hierarchy:
- Technical documentation
- API references
- User manuals
- Books
- Research papers with sections

### Limitations

- Requires **well-formatted source** (markdown headings, LaTeX sections, HTML headers)
- Sections can still be too long → combine with recursive splitting for large sections

---

## Strategy 4: Semantic Chunking via Embeddings ✅ (Meaning-Based)

### What It Does

1. Split text into sentences
2. Embed each sentence
3. Compute similarity between consecutive sentences
4. Group sentences where similarity stays above threshold
5. Start new chunk when similarity drops

```python
# Pseudocode
for i in range(1, len(sentences)):
    similarity = cosine_sim(embed[i-1], embed[i])
    if similarity < threshold:
        # Topic shift detected → new chunk
        start_new_chunk()
```

### Why It Works

- **Topic coherence** — chunks correspond to coherent topics
- **Automatic boundary detection** — finds topic shifts without manual rules
- **Language-agnostic** — works for any language with sentence embeddings
- **Handles unstructured text** — doesn't require headings or formatting

### Example

Input (three topics):
```
The API Gateway handles authentication. JWT tokens expire after 1 hour.
Rate limiting is enforced at 1000 req/min.

The database uses PostgreSQL for ACID transactions. We partition tables by date.
Backup retention is 7 years for compliance.

Security audits happen quarterly. PCI-DSS Level 1 compliance is maintained.
```

Output (3 chunks):
- **Chunk 1** (auth/rate limiting): "The API Gateway handles... 1000 req/min."
- **Chunk 2** (database): "The database uses PostgreSQL... 7 years for compliance."
- **Chunk 3** (security): "Security audits happen... maintained."

### When to Use

Best for:
- **Unstructured documents** (emails, chat logs, transcripts)
- **Topic segmentation** (news articles, blog posts)
- **Mixed-format documents** (poorly structured PDFs, OCR output)

### Limitations

- **Computationally expensive** — must embed every sentence
- **Threshold tuning** — similarity threshold is dataset-dependent
- **Embedding quality** — depends on quality of embedding model
- **Sentence segmentation** — struggles with domain-specific abbreviations (e.g., "Ph.D.")

---

## Strategy 5: Overlapping Context Windows ✅ (Continuity)

### What It Does
Take chunks from any strategy and add overlap from adjacent chunks.

```python
for i, chunk in enumerate(chunks):
    prev_overlap = chunks[i-1].page_content[-200:] if i > 0 else ""
    next_overlap = chunks[i+1].page_content[:200] if i < len(chunks)-1 else ""

    windowed_content = f"[PREV: ...{prev_overlap}]\n\n{chunk}\n\n[NEXT: {next_overlap}...]"
```

### Why It Works

**Problem it solves:**
```
Chunk N:   "...the system uses PostgreSQL."
Chunk N+1: "This choice was made because..."
           ^^^^^^ what is "this choice"?
```

**With overlap:**
```
Chunk N+1 with context:
[PREV CONTEXT: ...the system uses PostgreSQL.]

This choice was made because we needed ACID guarantees and complex query support.

[NEXT CONTEXT: PostgreSQL configuration includes...]
```

Now the LLM understands "this choice" = "PostgreSQL".

### When to Use

**Always add to any chunking strategy** when:
- Chunks reference previous context ("this approach", "the aforementioned method")
- Answers require understanding preceding/following chunks
- Multi-step reasoning across chunks

### Recommended Overlap Size
- **Small chunks (<500 chars):** 100-150 char overlap
- **Medium chunks (500-1500 chars):** 150-250 char overlap
- **Large chunks (>1500 chars):** 250-400 char overlap

**Rule of thumb:** 15-20% overlap to balance context vs. duplication.

---

## Strategy 6: Global Summary + Chunks ✅ (Document-Wide Context)

### What It Does
Create a high-level summary of the entire document and prepend it to every chunk.

```python
summary = summarize(full_document)  # via LLM or extractive

for chunk in chunks:
    augmented = f"{summary}\n\n---\n\nCHUNK:\n{chunk}"
```

### Why It Works

**Problem:**
```
Query: "What are the security implications of the payment system?"

Chunk 5 (from Chapter 3): "PCI-DSS Level 1 compliance is maintained..."
```

Without global context, the LLM doesn't know:
- What payment system?
- What architecture?
- What scale?

**With global summary:**
```
SUMMARY:
This document describes a distributed payment processing system handling millions
of daily transactions across three geographic regions. The architecture consists of
API Gateway, Processing Layer, and Data Layer with event sourcing and polyglot persistence.

---

CHUNK:
PCI-DSS Level 1 compliance is maintained. Cardholder data is tokenized immediately
at point of entry. No plaintext card numbers stored anywhere in the system.
```

Now the LLM has **document-wide context** for every chunk.

### When to Use

Essential for:
- **Multi-chapter documents** where chunks from different chapters are retrieved together
- **Technical architecture docs** where system overview is needed to understand details
- **Legal/policy documents** where preamble/definitions apply to all sections
- **Research papers** where abstract/intro provides context for methods/results

### How to Summarize

**Option 1: LLM summarization** (best quality)
```python
summary = llm.invoke("Summarize this document in 3-4 paragraphs:\n\n{full_text}")
```

**Option 2: Extractive summary** (fast, deterministic)
```python
# Executive summary + first paragraph + last paragraph
summary = extract_sections(full_text, ["Executive Summary", "Conclusion"])
```

**Option 3: Hierarchical summary**
```python
# Summarize each chapter, then summarize the chapter summaries
chapter_summaries = [summarize(ch) for ch in chapters]
global_summary = summarize(chapter_summaries)
```

### Cost Consideration

Adding a 500-token summary to **every chunk** increases token usage:
- 100 chunks × 500 summary tokens = **50K extra input tokens per query**
- At $3/1M input tokens (GPT-4), that's $0.15 per query
- For high-volume systems, consider:
  - Shorter summaries (200-300 tokens)
  - Selective summary (only add to chunks that need it)
  - Cache summaries (reuse across queries)

---

## Strategy 7: Embedding-Based Clustering ✅ (Multi-Document)

### What It Does
Cluster chunks by semantic similarity to identify related content across the document(s).

```python
embeddings = embed_all_chunks(chunks)
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(embeddings)

clusters = group_by_label(chunks, labels)
```

### Why It Works

**Use case:** "Find all chunks related to security, regardless of where they appear in the document."

```
Cluster 3 (Security):
- Chunk 15 (Chapter 1): "API Gateway authentication via JWT..."
- Chunk 47 (Chapter 3): "PCI-DSS compliance maintained..."
- Chunk 92 (Appendix): "Security audit procedures..."
```

All three chunks mention security but are scattered across the document. Clustering groups them together.

### When to Use

Best for:
- **Multi-document search** (e.g., "find all mentions of X across 100 documents")
- **Cross-referencing** ("show related chunks from different sections")
- **Topic extraction** ("what are the main topics in this corpus?")
- **Deduplication** (find near-duplicate chunks)

### Clustering Algorithms

| Algorithm | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| **KMeans** | Fast, scalable | Requires k | Known number of topics |
| **HDBSCAN** | Auto k, finds noise | Slower | Unknown number of topics |
| **Agglomerative** | Hierarchical clusters | O(n²) memory | Small datasets (<10K chunks) |

### Limitations

- **Doesn't replace retrieval** — still need vector search for query matching
- **Cluster quality** — depends on embedding model and number of clusters
- **Interpretability** — cluster labels must be inferred (via centroid summary)

---

## Comparison Matrix

| Strategy | Setup Complexity | Runtime Cost | Preserves Structure | Semantic Coherence | Best For |
|----------|-----------------|--------------|---------------------|-------------------|----------|
| Naive splitting | ★☆☆☆☆ | ★☆☆☆☆ | ❌ | ❌ | Demos only |
| Recursive splitting | ★★☆☆☆ | ★☆☆☆☆ | ✅ | ⚠️  | Default baseline |
| Layout-aware | ★★★☆☆ | ★☆☆☆☆ | ✅✅ | ✅ | Structured docs |
| Semantic (embedding) | ★★★☆☆ | ★★★☆☆ | ⚠️  | ✅✅ | Unstructured docs |
| Overlapping windows | ★☆☆☆☆ | ★★☆☆☆ | ✅ | ✅ | Add to any strategy |
| Global summary | ★★★★☆ | ★★★★☆ | ✅ | ✅✅ | Multi-chapter docs |
| Clustering | ★★★★☆ | ★★★☆☆ | ⚠️  | ✅✅ | Multi-doc search |

**Legend:**
- ★: Low/Simple
- ★★★★★: High/Complex
- ✅: Yes / Good
- ⚠️ : Partial / Depends
- ❌: No / Poor

---

## Production Recommendations

### For Most Use Cases (Start Here)

```python
# Step 1: Recursive splitting (respects structure)
base_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
)

# Step 2: Add overlapping context windows
chunks = add_overlap(base_splitter.split(text), overlap=200)

# Step 3: (Optional) Prepend global summary if document >5000 tokens
if len(text) > 5000:
    summary = summarize(text)
    chunks = [f"{summary}\n\n{chunk}" for chunk in chunks]
```

### For Structured Technical Documentation

```python
# Step 1: Split by headers
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Chapter"), ("##", "Section"), ("###", "Subsection")]
)

# Step 2: Split large sections further
chunks = []
for header_chunk in header_splitter.split(text):
    if len(header_chunk) > 1500:
        sub_chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split([header_chunk])
        # Preserve header metadata
        for sub in sub_chunks:
            sub.metadata.update(header_chunk.metadata)
        chunks.extend(sub_chunks)
    else:
        chunks.append(header_chunk)

# Step 3: Add global summary
summary = extract_summary(text)
chunks = [{**chunk, "global_context": summary} for chunk in chunks]
```

### For Unstructured Mixed-Format Documents

```python
# Step 1: Semantic chunking (embedding-based)
semantic_chunks = semantic_split(text, similarity_threshold=0.6)

# Step 2: Add overlap
chunks = add_overlap(semantic_chunks, overlap=150)

# Step 3: Cluster for multi-document search
embeddings = embed(chunks)
clusters = kmeans_cluster(embeddings, n_clusters=10)
```

---

## Evaluation Metrics

### How to Measure Chunking Quality

1. **Retrieval Precision**
   ```python
   # For each question, does the top-k retrieved chunks contain the answer?
   precision = correct_retrievals / total_queries
   ```

2. **Context Sufficiency**
   ```python
   # Can the LLM answer the question using ONLY the retrieved chunk?
   # (no need for additional context from other chunks)
   sufficiency = answerable_from_chunk_alone / total_queries
   ```

3. **Broken Structure Count**
   ```python
   # Count chunks with broken code blocks, tables, lists
   broken = count_broken_structures(chunks)
   ```

4. **Semantic Coherence** (human eval or LLM-as-judge)
   ```python
   # Does the chunk form a coherent, self-contained thought?
   coherence_score = llm_judge("Rate coherence 1-5", chunk)
   ```

### Benchmark Your Strategy

```python
test_questions = [
    "How does the system handle fraud detection?",
    "What database is used for transactions?",
    "Explain the API Gateway authentication flow.",
]

for strategy in [naive, recursive, layout_aware, semantic]:
    chunks = strategy.split(document)
    vectorstore = build_vectorstore(chunks)

    hits = 0
    for question in test_questions:
        results = vectorstore.similarity_search(question, k=3)
        if answer_found_in(results):
            hits += 1

    print(f"{strategy.__name__}: {hits}/{len(test_questions)} precision")
```

---

## Common Pitfalls

### 1. Chunk Size Too Small
```python
# BAD: 200-char chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200)
```
**Problem:** Fragments context, increases retrieval noise, loses semantic coherence.

**Fix:** Use 800-1500 char chunks for most documents.

### 2. No Overlap
```python
# BAD: No overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
```
**Problem:** Chunks that reference previous context ("this approach", "as mentioned above") lose meaning.

**Fix:** Use 15-20% overlap.

### 3. Ignoring Document Structure
```python
# BAD: Character splitting on a well-structured markdown document
CharacterTextSplitter(chunk_size=500)
```
**Problem:** Breaks headings, code blocks, lists.

**Fix:** Use `MarkdownHeaderTextSplitter` or `RecursiveCharacterTextSplitter`.

### 4. One-Size-Fits-All
```python
# BAD: Same chunking for code, prose, tables
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
```
**Problem:** Code needs smaller chunks (functions), prose can use larger chunks.

**Fix:** Detect content type and use appropriate strategy per section.

### 5. Not Testing Retrieval
```python
# BAD: Chunk and deploy without testing
chunks = splitter.split(text)
vectorstore = FAISS.from_documents(chunks)
# Ship it!
```
**Problem:** You don't know if retrieval actually works for your queries.

**Fix:** Create a test set of 20-30 questions, measure precision@k before deploying.

---

## Summary: Decision Tree

```
START: Do you have a long document to chunk?
│
├─ Is it well-structured (headings/sections)?
│  ├─ YES → Use MarkdownHeaderTextSplitter
│  │        + RecursiveCharacterTextSplitter for large sections
│  │        + Add global summary if multi-chapter
│  └─ NO  → Is it mostly unstructured (emails, transcripts)?
│           ├─ YES → Use semantic chunking (embedding-based)
│           └─ NO  → Use RecursiveCharacterTextSplitter (baseline)
│
├─ Do chunks reference previous context?
│  └─ YES → Add overlapping context windows (15-20% overlap)
│
├─ Is the document >5000 tokens?
│  └─ YES → Prepend global summary to each chunk
│
└─ Do you need to find related chunks across sections?
   └─ YES → Add embedding-based clustering
```

---

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Pinecone: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [OpenAI: Best Practices for Production RAG](https://platform.openai.com/docs/guides/production-best-practices)
- [Anthropic: Long Context Prompting Guide](https://docs.anthropic.com/claude/docs/long-context-prompting)
