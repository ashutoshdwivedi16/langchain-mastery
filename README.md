# 🦜🔗 langchain-mastery

> **A complete, hands-on LangChain learning journey** — from zero to building real AI applications.
> 100% free local setup available (no API keys needed!).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Learning](https://img.shields.io/badge/Learning-In%20Public-purple?style=flat-square)
![Day](https://img.shields.io/badge/90--Day_Challenge-Day%201-red?style=flat-square)

---

## 🗺️ Quick Navigation

| 📂 Section | 📄 What's Inside | 🔗 Link |
|-----------|-----------------|---------|
| **💻 Code** | All runnable examples & projects | [`code/`](./code/) |
| **📚 Docs** | Learning journal, concepts, guides | [`docs/`](./docs/) |
| **📖 Resources** | Books, courses, papers, references | [`resources/`](./resources/) |
| **📝 Journal** | Daily learning log (Week by week) | [`docs/journal/`](./docs/journal/) |
| **🧠 Concepts** | Deep-dive explanations | [`docs/concepts/`](./docs/concepts/) |
| **🚀 Quick Start** | Get running in 5 minutes | [QUICKSTART.md](./QUICKSTART.md) |

---

## 🌐 Learning Ecosystem

This is **Repository 1 of 5** in my 90-day public learning challenge:

| # | Repository | Focus | Status |
|---|-----------|-------|--------|
| 1 | 🦜 **langchain-mastery** ← *You are here* | LangChain, LLMs, RAG, Agents | 🟢 Active (Day 1/21) |
| 2 | 🐍 python-mastery | Python advanced patterns | 🔜 Day 22 |
| 3 | 🔥 pytorch-mastery | Deep learning & neural nets | 🔜 Day 50 |
| 4 | ☕ java-mastery | Java & Spring Boot | 🔜 Day 78 |
| 5 | 🏠 portfolio-showcase | Meta portfolio hub | 🔜 Ongoing |

---

## 🎯 What You'll Learn

```
Week 1 — Foundations
  ✅ LLM basics (local + API)
  ✅ Prompt templates & LCEL chains
  ✅ Output parsing (Pydantic, JSON)

Week 2 — Intermediate
  ✅ Memory types (buffer, window, summary)
  ✅ RAG with vector stores (FAISS, Chroma)
  ✅ Document loading & text splitting

Week 3 — Advanced + Projects
  ✅ Agents with custom tools
  ✅ Project: Personal Knowledge Base
  ✅ Project: Document Chatbot
  ✅ Project: Code Assistant
```

---

## 🚀 Quick Start (Choose Your Path)

### 🆓 Path A: Free Local Setup (No API Key!)
```bash
# 1. Install Ollama  →  https://ollama.ai
ollama pull llama2

# 2. Install dependencies
pip install -r code/requirements-local.txt

# 3. Run first example
python code/basics/01_basic_llm_local.py
```

### 💳 Path B: OpenAI API
```bash
# 1. Get API key  →  https://platform.openai.com
cp .env.example .env
# Add: OPENAI_API_KEY=sk-...

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Run first example
python code/basics/01_basic_llm.py
```

### 🎮 Interactive Mode
```bash
# Menu-driven exploration of all examples
python code/scripts/run_all_local.py
```

---

## 📁 Repository Structure

```
langchain-mastery/
│
├── 📋 README.md              ← You are here
├── 📋 QUICKSTART.md          ← 5-min onboarding
├── 📋 CONTRIBUTING.md        ← How to contribute
├── 📋 CHANGELOG.md           ← What changed when
├── 📋 LICENSE                ← MIT
│
├── 💻 code/                  ← ALL EXECUTABLE CODE
│   ├── requirements.txt      ← API version deps
│   ├── requirements-local.txt ← Free/local deps
│   ├── basics/               ← Week 1 examples (01–06)
│   ├── intermediate/         ← Week 2 examples (coming)
│   ├── advanced/             ← Week 3 examples (coming)
│   ├── projects/             ← Real applications (coming)
│   ├── templates/            ← Starter templates (coming)
│   └── scripts/              ← Setup & runners
│
├── 📚 docs/                  ← ALL DOCUMENTATION
│   ├── journal/              ← Daily learning log
│   │   ├── week-01/          ← This week's entries
│   │   └── ...
│   ├── concepts/             ← Deep-dive guides (coming)
│   ├── guides/               ← How-to tutorials (coming)
│   ├── comparisons/          ← Technology comparisons (coming)
│   └── diagrams/             ← Visual aids (coming)
│
└── 📖 resources/             ← REFERENCE MATERIALS
    ├── books/                ← Book notes (coming)
    ├── courses/              ← Course materials (coming)
    └── reference/            ← Cheat sheets (coming)
```

---

## 💻 Code Examples

### Basics (`code/basics/`)

| File | Topic | API | Local (Free) |
|------|-------|-----|-------------|
| `01_basic_llm.py` | Simple LLM calls | ✅ | — |
| `01_basic_llm_local.py` | LLM with Ollama | — | ✅ |
| `02_prompts_and_chains.py` | Templates & LCEL | ✅ | — |
| `03_memory_and_conversation.py` | Memory types | ✅ | — |
| `04_rag_vector_stores.py` | RAG + FAISS/Chroma | ✅ | — |
| `04_rag_vector_stores_local.py` | RAG + HuggingFace | — | ✅ |
| `05_agents_and_tools.py` | Agents & tools | ✅ | — |
| `06_output_parsers.py` | Pydantic, JSON | ✅ | — |

---

## 📊 Progress Tracker

### Week 1: LangChain Foundations
| Day | Topic | Status | Journal |
|-----|-------|--------|---------|
| 1 | Setup & Structure | ✅ Done | [Day 1](./docs/journal/week-01/day-01-setup.md) |
| 2 | Basic LLM | 🔜 Tomorrow | — |
| 3 | Prompts & Chains | ⏳ Pending | — |
| 4 | Output Parsers | ⏳ Pending | — |
| 5 | Memory | ⏳ Pending | — |
| 6 | RAG | ⏳ Pending | — |
| 7 | Reflection | ⏳ Pending | — |

---

## 🆚 Free vs API Comparison

| Feature | 🆓 Local (Ollama) | 💳 API (OpenAI) |
|---------|------------------|-----------------|
| Cost | **$0 forever** | ~$0.002 / 1K tokens |
| Privacy | **100% local** | Sent to OpenAI |
| Setup | 5 minutes | Need credit card |
| Quality | Good (llama2/mistral) | Excellent (GPT-4) |
| Speed | Depends on hardware | Very fast |
| Internet | Not needed* | Required |

*After initial model download*

---

## 🤝 Contributing

Found a bug? Have an improvement? See [CONTRIBUTING.md](./CONTRIBUTING.md).

All skill levels welcome — this is a learning repo! 🎓

---

## 📜 License

MIT — free to use, share, and build upon. See [LICENSE](./LICENSE).

---

<div align="center">

**Built in public, one commit at a time 🚀**

*Part of a 90-day learning challenge*

⭐ Star this repo if it helped you learn!

</div>
