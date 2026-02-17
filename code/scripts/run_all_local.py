"""
LangChain Hello World - LOCAL VERSION (No API Required)
Run all examples using free local tools
"""

import sys
import importlib
import subprocess

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print(" " * 10 + "🦜 LANGCHAIN HELLO WORLD - LOCAL EDITION 🔗")
    print(" " * 12 + "100% Free - No API Keys Required!")
    print("="*70)

def check_prerequisites():
    """Check if required tools are installed"""
    print("\n🔍 Checking Prerequisites...")
    print("="*70)

    issues = []

    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Ollama installed:", result.stdout.strip())
        else:
            issues.append("Ollama")
    except FileNotFoundError:
        issues.append("Ollama")
        print("❌ Ollama not found")

    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get('models', [])
            if models:
                print(f"✅ Available models: {len(models)}")
                for model in models:
                    print(f"   - {model['name']}")
            else:
                issues.append("No models downloaded")
                print("⚠️  No models found. Run: ollama pull llama2")
        else:
            issues.append("Ollama not running")
            print("⚠️  Ollama not responding")
    except Exception as e:
        issues.append("Ollama not running")
        print("❌ Ollama not running. Start with: ollama serve")

    # Check Python packages
    required_packages = [
        ('langchain', 'langchain'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu')
    ]

    for package_name, install_name in required_packages:
        try:
            __import__(package_name)
            print(f"✅ {install_name} installed")
        except ImportError:
            issues.append(f"{install_name} package")
            print(f"❌ {install_name} not installed")

    if issues:
        print("\n" + "="*70)
        print("⚠️  Issues Found:")
        for issue in issues:
            print(f"   - {issue}")

        print("\n📚 Setup Instructions:")
        print("   1. Install Ollama: python 00_setup_local.py")
        print("   2. Install packages: pip install -r requirements-local.txt")
        print("   3. Start Ollama: ollama serve")
        print("   4. Download a model: ollama pull llama2")
        print("="*70)

        return False

    print("\n✅ All prerequisites met!")
    print("="*70)
    return True

def print_menu():
    """Display feature menu"""
    print("\n📋 Available Features (Local/Free):")
    print("="*70)
    print("  1. Basic LLM Interaction (Ollama)")
    print("  2. RAG with Vector Stores (HuggingFace + FAISS)")
    print("  3. View Setup Guide")
    print("  4. Check System Status")
    print("  0. Exit")
    print("="*70)
    print("\n💡 Note: Some advanced features (agents, complex chains) work")
    print("   better with API-based models, but basics work great locally!")

def run_feature(feature_number):
    """Run a specific feature example"""
    feature_map = {
        1: ("01_basic_llm_local", "Basic LLM with Ollama"),
        2: ("04_rag_vector_stores_local", "RAG with Local Tools"),
    }

    if feature_number not in feature_map:
        print(f"❌ Invalid feature number: {feature_number}")
        return False

    module_name, feature_name = feature_map[feature_number]

    print(f"\n{'='*70}")
    print(f"🚀 Running: {feature_name}")
    print('='*70)

    try:
        module = importlib.import_module(module_name)
        module.main()
        print(f"\n✅ Completed: {feature_name}")
        return True
    except Exception as e:
        print(f"\n❌ Error running {feature_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_setup_guide():
    """Display setup guide"""
    print("\n" + "="*70)
    print("📖 SETUP GUIDE - Running LangChain Without API Keys")
    print("="*70)

    print("""
🎯 What You Need:

1. **Ollama** (Free Local LLM)
   - Download: https://ollama.ai/download
   - macOS: brew install ollama
   - Linux: curl -fsSL https://ollama.ai/install.sh | sh
   - Windows: Download installer from website

2. **Python Packages**
   pip install -r requirements-local.txt

3. **Start Ollama**
   ollama serve

4. **Download a Model**
   ollama pull llama2        # Good all-around model (3.8GB)
   ollama pull mistral       # Fast and efficient (4.1GB)
   ollama pull phi           # Lightweight for testing (1.6GB)

📚 What Works Locally:

✅ Basic LLM interaction
✅ Prompt templates and chains
✅ RAG with vector stores
✅ Text splitting and embeddings
✅ Memory and conversations
✅ Simple agents (with limitations)

⚠️  What's Limited Locally:

⚡ Complex agent reasoning (works better with GPT-4)
⚡ Very long context windows
⚡ Specialized domain knowledge

💰 Cost Comparison:

Ollama (Local):
  - Cost: $0 forever
  - Speed: Depends on your hardware
  - Privacy: 100% local, your data never leaves

OpenAI API:
  - Cost: ~$0.002 per 1K tokens
  - Speed: Very fast
  - Privacy: Data sent to OpenAI

🚀 Getting Started:

Step 1: Run the setup script
    python 00_setup_local.py

Step 2: Verify everything works
    python 01_basic_llm_local.py

Step 3: Try RAG
    python 04_rag_vector_stores_local.py

Step 4: Build your own app!
""")

    print("="*70)

def check_status():
    """Check system status"""
    print("\n" + "="*70)
    print("🔍 System Status Check")
    print("="*70)

    check_prerequisites()

    print("\n💻 System Info:")
    import platform
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("   PyTorch: Not installed")

    print("="*70)

def interactive_mode():
    """Run in interactive mode with menu"""
    while True:
        print_menu()
        try:
            choice = input("\n👉 Select an option (0-4): ").strip()

            if choice == '0':
                print("\n👋 Thanks for exploring LangChain locally! Happy coding!")
                break
            elif choice == '1':
                run_feature(1)
            elif choice == '2':
                run_feature(2)
            elif choice == '3':
                show_setup_guide()
            elif choice == '4':
                check_status()
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 4.")

            if choice != '0':
                print("\n" + "="*70)
                input("Press Enter to return to menu...")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main entry point"""
    print_banner()

    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠️  Setup incomplete. Would you like to see the setup guide? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            show_setup_guide()
        return

    # Check for command-line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "setup":
            show_setup_guide()
        elif arg == "status":
            check_status()
        elif arg.isdigit() and 1 <= int(arg) <= 2:
            run_feature(int(arg))
        else:
            print(f"❌ Invalid argument: {arg}")
            print("\nUsage:")
            print("  python run_all_local.py         # Interactive mode")
            print("  python run_all_local.py setup   # Show setup guide")
            print("  python run_all_local.py status  # Check system status")
            print("  python run_all_local.py 1       # Run feature 1")
            print("  python run_all_local.py 2       # Run feature 2")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
