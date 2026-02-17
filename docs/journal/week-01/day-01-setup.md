# Day 1: Project Setup & Goals 🚀

**Date:** 2025-02-17
**Time Spent:** ~2 hours
**Week:** 1 of 3 (LangChain Mastery)
**Commits:** Initial setup commit

---

## 🎯 Goals for Today

- [x] Create repository structure (`code/`, `docs/`, `resources/`)
- [x] Organize existing LangChain examples
- [x] Set up git & push to GitHub
- [x] Write this journal entry
- [x] Define goals for the 90-day journey

---

## 🌟 Why I'm Doing This

I want to learn LangChain deeply — not just follow tutorials, but actually **understand** how to build AI-powered applications. The best way I know to learn is:

1. **Build things** that actually run
2. **Write about what I'm learning** to solidify understanding
3. **Learn in public** so I'm accountable and others can benefit

This repo is part of a bigger 90-day challenge where I'll master LangChain, Python patterns, PyTorch, and Java — all with documented learning journals.

---

## 📋 90-Day Learning Plan Overview

| Repository | Focus | Days |
|-----------|-------|------|
| `langchain-mastery` | LangChain + LLMs + RAG + Agents | 1–21 |
| `python-mastery` | Advanced Python patterns | 22–49 |
| `pytorch-mastery` | Deep learning & neural networks | 50–77 |
| `java-mastery` | Java + Spring Boot | 78–90 |

---

## 🏗️ What I Built Today

### Repository Structure

```
langchain-mastery/
├── code/
│   ├── basics/      ← 8 example files moved here
│   └── scripts/     ← runner & setup scripts
├── docs/
│   └── journal/     ← This file!
└── resources/       ← for books, courses, references
```

### Files Organized

**`code/basics/` — Core LangChain examples:**
- `01_basic_llm.py` — Simple LLM calls via OpenAI
- `01_basic_llm_local.py` — Same but free with Ollama
- `02_prompts_and_chains.py` — Prompt templates + LCEL
- `03_memory_and_conversation.py` — 3 memory types
- `04_rag_vector_stores.py` — RAG with FAISS & Chroma
- `04_rag_vector_stores_local.py` — RAG for free with HuggingFace
- `05_agents_and_tools.py` — React agents + custom tools
- `06_output_parsers.py` — Structured outputs with Pydantic

---

## 🎓 What I Know About LangChain So Far

Before today, I had already built examples for these concepts:

### Key Concepts I've Touched:

**1. LLM Abstraction**
LangChain wraps different LLM providers (OpenAI, Ollama, HuggingFace) behind a unified interface. You can swap models by changing one line.

**2. LCEL (LangChain Expression Language)**
The `|` pipe syntax chains components together:
```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": "hello"})
```
This is elegant — very similar to Unix pipes.

**3. Prompt Templates**
Instead of string formatting, you define reusable templates:
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{question}")
])
```

**4. Memory**
Three main types I've learned:
- `ConversationBufferMemory` — stores everything
- `ConversationBufferWindowMemory` — keeps last K turns
- `ConversationSummaryMemory` — compresses history

**5. RAG**
The pattern: `Documents → Embeddings → Vector Store → Retriever → LLM`
Works 100% locally with HuggingFace + FAISS.

**6. Agents**
Agents decide *which tools to call* and *in what order*. The React pattern is: Reason → Act → Observe → Repeat.

---

## 🐛 Challenges Faced Today

**Challenge 1: No API Key**
- **Problem:** Most LangChain tutorials assume OpenAI access
- **Solution:** Found Ollama — runs LLMs locally for free!
- **Insight:** The local path actually teaches you more because you understand the infrastructure

**Challenge 2: Organizing files**
- **Problem:** Flat file structure gets messy fast
- **Solution:** Adopted `code/basics/`, `docs/`, `resources/` pattern
- **Insight:** Good organization from day 1 pays off later

---

## 💡 Aha Moments

> **"LCEL chains are just function composition."**

When I realized `prompt | llm | parser` is the same as `parser(llm(prompt(input)))`, it clicked. LangChain isn't magic — it's just a clean API for composing AI operations.

> **"Local LLMs are a superpower for learning."**

No API costs = no fear of experimenting. I can run 100 queries to understand how temperature affects output without worrying about bills.

---

## 📚 Resources Bookmarked

- [LangChain Official Docs](https://python.langchain.com/)
- [Ollama Model Library](https://ollama.ai/library)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [HuggingFace Sentence Transformers](https://sbert.net/)

---

## 🔄 What I'd Do Differently

- Set up the directory structure **first** before writing any code
- Add docstrings to all functions from the beginning
- Write journal entries **during** learning, not after

---

## ⭐ Wins of the Day

- ✅ Repository organized and on GitHub
- ✅ 8 working LangChain examples documented
- ✅ Dual learning path: API + completely free local
- ✅ First journal entry written
- ✅ **Day 1 streak started!** 🔥

---

## ⏭️ Tomorrow — Day 2: Basic LLM

**Plan:**
1. Run `01_basic_llm_local.py` with Ollama
2. Experiment with llama2, mistral, and phi models
3. Compare outputs at different temperatures (0.0, 0.5, 0.9, 1.5)
4. Document differences in a concept guide
5. Write Day 2 journal

**Question to explore:** How does temperature actually affect output quality and creativity?

---

*Day 1 complete. The journey begins.* 🚀
