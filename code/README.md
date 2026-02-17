# 💻 Code

All runnable LangChain examples, organized by difficulty.

---

## 📂 Structure

```
code/
├── requirements.txt         ← OpenAI API version deps
├── requirements-local.txt   ← Free/local (Ollama) deps
├── basics/                  ← Week 1: Foundation examples
├── intermediate/            ← Week 2: Coming soon
├── advanced/                ← Week 3: Coming soon
├── projects/                ← Real applications: Coming soon
├── templates/               ← Starter templates: Coming soon
└── scripts/                 ← Setup & runner utilities
```

---

## 🚀 Setup

### Free Local Path (Recommended for beginners)
```bash
# 1. Install Ollama
#    macOS: brew install ollama
#    Linux: curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download a model
ollama pull llama2

# 3. Install Python deps
pip install -r requirements-local.txt

# 4. Run!
python basics/01_basic_llm_local.py
```

### API Path (Requires OpenAI account)
```bash
# 1. Set up API key
export OPENAI_API_KEY=sk-...

# 2. Install deps
pip install -r requirements.txt

# 3. Run!
python basics/01_basic_llm.py
```

---

## 📋 Examples Index

### basics/ — Week 1 Foundation

| # | File | Concept | Path | Free? |
|---|------|---------|------|-------|
| 1a | `01_basic_llm.py` | LLM calls (API) | basics/ | ❌ |
| 1b | `01_basic_llm_local.py` | LLM with Ollama | basics/ | ✅ |
| 2 | `02_prompts_and_chains.py` | Templates + LCEL | basics/ | ❌ |
| 3 | `03_memory_and_conversation.py` | Memory types | basics/ | ❌ |
| 4a | `04_rag_vector_stores.py` | RAG + FAISS | basics/ | ❌ |
| 4b | `04_rag_vector_stores_local.py` | RAG + HuggingFace | basics/ | ✅ |
| 5 | `05_agents_and_tools.py` | Agents & tools | basics/ | ❌ |
| 6 | `06_output_parsers.py` | Pydantic, JSON | basics/ | ❌ |

### intermediate/ — Week 2 *(Coming Day 8)*
### advanced/ — Week 3 *(Coming Day 15)*
### projects/ — Real Apps *(Coming Day 17)*

---

## 🎮 Interactive Runner

```bash
# Menu-driven interface to run any example
python scripts/run_all_local.py

# Or run a specific example
python scripts/run_all_local.py 1
```

---

*New examples added daily during the 21-day sprint.*
