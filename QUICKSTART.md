# 🚀 Quick Start Guide - No API Version

**Get started with LangChain in 5 minutes - completely FREE!**

## ⚡ Super Quick Setup

### 1. Install Ollama (Pick your OS)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
- Go to: https://ollama.ai/download
- Download and run installer

### 2. Install Python Requirements

```bash
cd langchain-hello-world
pip install -r requirements-local.txt
```

### 3. Run Automated Setup

```bash
python 00_setup_local.py
```

This will:
- ✅ Check if Ollama is installed
- ✅ Help you download a model
- ✅ Create configuration files
- ✅ Verify everything works

### 4. Start Exploring!

```bash
python run_all_local.py
```

## 📖 Your First LangChain Program

Create a file called `my_first_langchain.py`:

```python
from langchain_community.llms import Ollama

# Initialize local LLM (no API key needed!)
llm = Ollama(model="llama2")

# Ask a question
response = llm.invoke("Explain LangChain in one sentence.")

print(response)
```

Run it:
```bash
python my_first_langchain.py
```

**That's it!** You just ran your first LangChain program with a local LLM! 🎉

## 🎯 What to Try Next

### Try Different Models

```bash
# Download other models
ollama pull mistral    # Fast and efficient
ollama pull phi        # Lightweight
ollama pull codellama  # Great for code
```

Use them in your code:
```python
llm = Ollama(model="mistral")
```

### Build a Simple Chatbot

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

print("Chatbot ready! (type 'quit' to exit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit':
        break

    response = llm.invoke(user_input)
    print(f"Bot: {response}")
```

### Try RAG (Document Q&A)

```bash
python 04_rag_vector_stores_local.py
```

This shows you how to:
- Load documents
- Create embeddings
- Search for relevant content
- Answer questions based on your documents

## 💡 Common Commands

```bash
# Check what models you have
ollama list

# Download a model
ollama pull llama2

# Start Ollama server (if not auto-started)
ollama serve

# Check system status
python run_all_local.py status

# View setup guide
python run_all_local.py setup

# Run basic LLM example
python 01_basic_llm_local.py

# Run RAG example
python 04_rag_vector_stores_local.py
```

## ❓ Troubleshooting

### "Ollama not running"
```bash
# Start Ollama in a new terminal
ollama serve
```

### "No models found"
```bash
# Download a model
ollama pull llama2
```

### "Import error"
```bash
# Install requirements
pip install -r requirements-local.txt
```

### Still stuck?
```bash
# Run diagnostics
python run_all_local.py status
```

## 🎓 Learning Path

1. ✅ **You are here** - Setup complete!
2. 📝 Run `01_basic_llm_local.py` - Learn basic LLM interaction
3. 🔍 Run `04_rag_vector_stores_local.py` - Learn RAG
4. 🛠️ Read the code and experiment!
5. 🚀 Build your own application!

## 📚 More Resources

- **Full Documentation:** See `README-LOCAL.md`
- **Original API Version:** See `README.md`
- **Ollama Docs:** https://ollama.ai
- **LangChain Docs:** https://python.langchain.com

---

**Questions?** Check `README-LOCAL.md` for detailed info!

**Happy Coding! 🎉**
