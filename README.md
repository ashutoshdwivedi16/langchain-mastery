# langchain-mastery

Production-ready LangChain patterns and implementations — covering the full stack from LLM abstraction to RAG pipelines, agents, and structured output. Supports both OpenAI-compatible APIs and fully local inference via Ollama.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-1C3C3C?style=flat-square)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

---

## Overview

This repository provides a structured reference implementation of core LangChain components. Each module is self-contained, runnable, and annotated to demonstrate both the mechanics and the reasoning behind each pattern.

**Supports two inference backends:**

| Backend | Setup | Cost | Privacy |
|--------|-------|------|---------|
| **Ollama** (local) | `ollama pull llama2` | Free | 100% local |
| **OpenAI API** | Set `OPENAI_API_KEY` | Pay-per-token | Cloud |

The local path uses `sentence-transformers` for embeddings and `FAISS` for vector storage — no external dependencies beyond the initial model download.

---

## Repository Structure

```
langchain-mastery/
├── code/
│   ├── basics/                  # Core pattern implementations
│   │   ├── 01_basic_llm.py                  # LLM abstraction layer
│   │   ├── 01_basic_llm_local.py            # Ollama backend
│   │   ├── 02_prompts_and_chains.py         # LCEL chain composition
│   │   ├── 03_memory_and_conversation.py    # Stateful conversation patterns
│   │   ├── 04_rag_vector_stores.py          # RAG pipeline (OpenAI + Chroma/FAISS)
│   │   ├── 04_rag_vector_stores_local.py    # RAG pipeline (local, zero-cost)
│   │   ├── 05_agents_and_tools.py           # ReAct agent + tool definitions
│   │   └── 06_output_parsers.py             # Pydantic, structured, JSON output
│   ├── intermediate/            # Advanced chain patterns
│   ├── advanced/                # Production patterns, async, streaming
│   ├── projects/                # End-to-end application implementations
│   ├── scripts/                 # CLI runners and setup utilities
│   ├── requirements.txt         # OpenAI backend dependencies
│   └── requirements-local.txt  # Local/Ollama backend dependencies
├── docs/
│   ├── journal/                 # Implementation notes and decisions
│   ├── concepts/                # Architecture deep-dives
│   ├── guides/                  # Deployment and integration guides
│   └── comparisons/             # Backend and model comparisons
└── resources/
    ├── books/
    ├── papers/
    └── reference/
```

---

## Installation

### Local Backend (Ollama)

```bash
# Install Ollama — https://ollama.ai
ollama pull llama2        # or: mistral, phi, codellama

git clone https://github.com/ashutoshdwivedi16/langchain-mastery.git
cd langchain-mastery
pip install -r code/requirements-local.txt
```

### OpenAI Backend

```bash
git clone https://github.com/ashutoshdwivedi16/langchain-mastery.git
cd langchain-mastery
pip install -r code/requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

---

## Usage

Run any module directly:

```bash
python code/basics/01_basic_llm_local.py
python code/basics/04_rag_vector_stores_local.py
python code/basics/05_agents_and_tools.py
```

Or use the interactive CLI:

```bash
python code/scripts/run_all_local.py
```

---

## Modules

### `01` — LLM Abstraction

Demonstrates the unified LangChain LLM interface across backends. Covers synchronous invocation, batch processing, and streaming callbacks.

```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2", temperature=0.7)
response = llm.invoke("Explain vector embeddings in two sentences.")
```

### `02` — Prompt Templates & LCEL Chains

LCEL pipe syntax for composing prompt → model → parser pipelines. Covers `PromptTemplate`, `ChatPromptTemplate`, and sequential chain patterns.

```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "retrieval-augmented generation"})
```

### `03` — Memory & Conversation State

Three memory strategies with different token-efficiency tradeoffs:
- `ConversationBufferMemory` — full history retention
- `ConversationBufferWindowMemory` — sliding window (last *k* turns)
- `ConversationSummaryMemory` — LLM-compressed history

### `04` — RAG Pipeline

Full retrieval-augmented generation implementation:

```
Documents → TextSplitter → Embeddings → VectorStore → Retriever → LLM
```

Local variant uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim) with FAISS for ANN search. Supports similarity search with relevance scores.

### `05` — Agents & Tools

ReAct agent with custom tool definitions using the `@tool` decorator. Demonstrates multi-step reasoning, tool chaining, and agent memory integration.

```python
@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}))
```

### `06` — Output Parsers

Structured extraction from LLM output:
- `CommaSeparatedListOutputParser`
- `StructuredOutputParser` with `ResponseSchema`
- `PydanticOutputParser` with validation and nested models

---

## Architecture Notes

**LCEL vs Legacy Chains**

All implementations use the LangChain Expression Language (LCEL) rather than legacy `LLMChain` / `SequentialChain`. LCEL provides native streaming, async support, and better composability.

**Embedding Model Selection**

For local deployments, `all-MiniLM-L6-v2` balances accuracy and speed (384 dimensions, ~80MB). For higher accuracy at larger scale, swap to `all-mpnet-base-v2` (768 dimensions) or a domain-specific model.

**Vector Store Tradeoffs**

| Store | Best For | Persistence |
|-------|----------|-------------|
| FAISS | In-process, high throughput | Manual (`save_local`) |
| Chroma | Persistent dev/test | Automatic |
| Pinecone | Production scale | Managed |

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Bug reports, improvements, and additional pattern implementations are welcome.

---

## License

MIT. See [LICENSE](./LICENSE).
