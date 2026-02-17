# 🦜🔗 LangChain Hello World - LOCAL EDITION (No API Required!)

**Learn LangChain 100% FREE - No API Keys, No Costs!**

This version uses completely free, local tools so you can learn and build with LangChain without spending a penny.

## 🎯 What Makes This Free?

| Component | Free Alternative | Why It's Great |
|-----------|-----------------|----------------|
| **LLM** | Ollama (Llama2, Mistral, Phi) | Runs on your computer, 100% private |
| **Embeddings** | HuggingFace Sentence Transformers | Open-source, works offline |
| **Vector DB** | FAISS / Chroma | Free, fast, local storage |
| **Cost** | **$0 forever** | No API calls, no subscriptions |

## 🚀 Quick Start (3 Steps)

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai/download](https://ollama.ai/download)

### Step 2: Set Up Python Environment

```bash
# Navigate to project
cd langchain-hello-world

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-local.txt
```

### Step 3: Run Setup & Examples

```bash
# Automated setup (downloads models, checks installation)
python 00_setup_local.py

# Start exploring!
python run_all_local.py
```

## 📋 What's Included

### Available Examples

1. **Basic LLM Interaction** (`01_basic_llm_local.py`)
   - Simple questions and answers
   - Streaming responses
   - Multiple queries
   - Uses: Ollama (Llama2/Mistral/Phi)

2. **RAG with Vector Stores** (`04_rag_vector_stores_local.py`)
   - Document embeddings (HuggingFace)
   - Semantic search (FAISS)
   - Question answering over documents
   - 100% local RAG system

3. **More Coming Soon!**
   - Prompt templates with local LLMs
   - Memory and conversations
   - Simple agents

## 🎮 Usage Examples

### Interactive Mode
```bash
python run_all_local.py
```

Shows a menu where you can:
- Run individual features
- Check system status
- View setup guide
- Test your installation

### Run Specific Examples
```bash
# Basic LLM
python 01_basic_llm_local.py

# RAG System
python 04_rag_vector_stores_local.py

# Check status
python run_all_local.py status

# Show setup guide
python run_all_local.py setup
```

## 🔧 Available Models

### Download Models
```bash
# Recommended: Good all-around model
ollama pull llama2          # 3.8GB

# Fast and efficient
ollama pull mistral         # 4.1GB

# Lightweight for testing
ollama pull phi             # 1.6GB

# Others
ollama pull codellama       # Great for code
ollama pull neural-chat     # Good for conversations
```

### List Installed Models
```bash
ollama list
```

### Switch Models in Code
```python
# Use Mistral instead of Llama2
llm = Ollama(model="mistral")

# Use Phi (lighter, faster)
llm = Ollama(model="phi")
```

## 💡 Learning Path

### Absolute Beginner
1. Run setup: `python 00_setup_local.py`
2. Try basic LLM: `python 01_basic_llm_local.py`
3. Explore the code and modify prompts

### Intermediate
1. Understand RAG: `python 04_rag_vector_stores_local.py`
2. Try different embedding models
3. Add your own documents

### Advanced
1. Combine RAG with custom prompts
2. Build a chatbot with memory
3. Create domain-specific applications

## 🆚 Local vs API Comparison

### Local (Ollama) ✅
- **Cost:** $0 forever
- **Privacy:** 100% - data never leaves your computer
- **Speed:** Depends on hardware (good on modern laptops)
- **Limits:** None! Use as much as you want
- **Internet:** Not needed after setup
- **Best for:** Learning, prototyping, privacy-sensitive apps

### API (OpenAI) ⚡
- **Cost:** ~$0.002 per 1K tokens (can add up)
- **Privacy:** Data sent to OpenAI
- **Speed:** Very fast
- **Limits:** Rate limits, quotas
- **Internet:** Required
- **Best for:** Production apps needing best quality

## 🎓 Features Comparison

| Feature | Works Locally? | Quality |
|---------|---------------|---------|
| Basic LLM calls | ✅ Perfect | Good |
| Prompt templates | ✅ Perfect | Good |
| Chains | ✅ Perfect | Good |
| Memory | ✅ Perfect | Good |
| RAG / Vector search | ✅ Perfect | Excellent |
| Simple agents | ✅ Works | Fair |
| Complex reasoning | ⚠️ Limited | Fair |
| Code generation | ✅ Good (use codellama) | Good |

## 🛠️ Troubleshooting

### Ollama Not Starting
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start manually
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Download missing model
ollama pull llama2
```

### Slow Performance
- Use smaller models (phi)
- Reduce temperature
- Use shorter prompts
- Close other applications

### Out of Memory
```bash
# Use lighter model
ollama pull phi

# Or reduce context in code
llm = Ollama(model="phi", num_ctx=2048)
```

## 📊 Hardware Requirements

### Minimum
- **RAM:** 8GB
- **Storage:** 5GB free
- **CPU:** Any modern processor
- **GPU:** Not required (but helps)

### Recommended
- **RAM:** 16GB+
- **Storage:** 10GB+ free
- **CPU:** Multi-core processor
- **GPU:** Optional (NVIDIA for acceleration)

### Model Sizes
- Phi: ~1.6GB (lightest)
- Llama2: ~3.8GB (balanced)
- Mistral: ~4.1GB (good quality)
- Llama2 70B: ~40GB (best quality, needs powerful hardware)

## 🎯 Next Steps

Once you're comfortable with the basics:

1. **Build a Personal Assistant**
   ```python
   # Chatbot that remembers conversations
   # Answers questions about your documents
   # Runs completely offline
   ```

2. **Create a Document Q&A System**
   ```python
   # Upload your PDFs, docs, text files
   # Ask questions and get answers
   # All data stays local
   ```

3. **Code Helper**
   ```python
   # Use codellama model
   # Get code suggestions
   # Debug and explain code
   ```

## 📚 Additional Resources

### Ollama
- Website: [ollama.ai](https://ollama.ai)
- Models: [ollama.ai/library](https://ollama.ai/library)
- GitHub: [github.com/ollama/ollama](https://github.com/ollama/ollama)

### LangChain
- Docs: [python.langchain.com](https://python.langchain.com)
- Community: [LangChain GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

### HuggingFace
- Models: [huggingface.co/models](https://huggingface.co/models)
- Sentence Transformers: [sbert.net](https://sbert.net)

## 🔐 Privacy & Security

### Why Local Matters
- **Your data stays on your computer**
- **No logs sent to third parties**
- **No internet required after setup**
- **Perfect for sensitive documents**
- **Compliant with data regulations**

### Use Cases
- Medical/legal document analysis
- Personal note organization
- Confidential business data
- Learning and experimentation
- Offline applications

## 🤝 Contributing

Try the examples and share what you build! Ideas for contributions:
- More example use cases
- Performance optimizations
- Additional model configurations
- Documentation improvements

## ❓ FAQ

**Q: Is this really free?**
A: Yes! Ollama and all tools used are completely free and open-source.

**Q: How does quality compare to ChatGPT?**
A: Local models are good but not as advanced as GPT-4. Great for most tasks!

**Q: Can I use this for commercial projects?**
A: Yes! Check individual model licenses, but most are permissive.

**Q: Do I need a GPU?**
A: No, but it helps with speed. Works fine on CPU.

**Q: Can I switch to OpenAI later?**
A: Yes! Just change the LLM initialization. Code stays the same.

**Q: What if I run into errors?**
A: Run `python run_all_local.py status` to diagnose issues.

---

## 🎉 Ready to Start?

```bash
# One command to set everything up
python 00_setup_local.py

# Then start exploring
python run_all_local.py
```

**Happy Learning! 🚀**

Built with ❤️ using free, open-source tools
