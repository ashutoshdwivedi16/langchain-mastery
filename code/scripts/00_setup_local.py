"""
Setup Script for Local LangChain (No API Required)
This script helps you set up local LLMs using Ollama
"""

import subprocess
import sys
import platform

def print_banner():
    print("\n" + "="*70)
    print(" " * 15 + "🦜 LANGCHAIN LOCAL SETUP 🔗")
    print(" " * 10 + "Run LangChain Without API Keys!")
    print("="*70)

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Ollama is installed:", result.stdout.strip())
            return True
        return False
    except FileNotFoundError:
        return False

def install_ollama_instructions():
    """Provide installation instructions for Ollama"""
    os_type = platform.system()

    print("\n📦 Ollama Installation Instructions:")
    print("="*70)

    if os_type == "Darwin":  # macOS
        print("\nFor macOS:")
        print("1. Visit: https://ollama.ai/download")
        print("2. Download Ollama for macOS")
        print("3. Install the downloaded app")
        print("\nOr use Homebrew:")
        print("   brew install ollama")

    elif os_type == "Linux":
        print("\nFor Linux:")
        print("Run this command:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")

    elif os_type == "Windows":
        print("\nFor Windows:")
        print("1. Visit: https://ollama.ai/download")
        print("2. Download Ollama for Windows")
        print("3. Run the installer")

    print("\n" + "="*70)

def pull_ollama_models():
    """Pull recommended Ollama models"""
    print("\n📥 Downloading Recommended Models:")
    print("="*70)

    models = [
        ("llama2", "General purpose, good for learning (3.8GB)"),
        ("mistral", "Fast and efficient (4.1GB)"),
        ("phi", "Lightweight, perfect for testing (1.6GB)")
    ]

    print("\nRecommended models:")
    for i, (model, desc) in enumerate(models, 1):
        print(f"{i}. {model} - {desc}")

    print("\n0. Skip for now")

    choice = input("\nSelect model to download (0-3): ").strip()

    if choice == "0":
        print("⏭️  Skipping model download")
        return

    if choice.isdigit() and 1 <= int(choice) <= len(models):
        model_name = models[int(choice) - 1][0]
        print(f"\n📥 Downloading {model_name}... (this may take a few minutes)")

        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"✅ Successfully downloaded {model_name}!")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to download {model_name}")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
    else:
        print("❌ Invalid choice")

def list_available_models():
    """List locally available Ollama models"""
    print("\n📋 Checking Available Local Models:")
    print("="*70)

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)

        if "NAME" in result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print("✅ You have models installed and ready to use!")
                return True

        print("⚠️  No models found. Please download a model first.")
        return False

    except subprocess.CalledProcessError:
        print("❌ Error checking models")
        return False
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False

def test_ollama_connection():
    """Test if Ollama is running"""
    print("\n🧪 Testing Ollama Connection:")
    print("="*70)

    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running and accessible!")
            return True
        else:
            print("⚠️  Ollama is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        print("\nTo start Ollama, run:")
        print("   ollama serve")
        print("\nOr on macOS/Windows, the Ollama app should auto-start")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def create_env_file():
    """Create .env file for local setup"""
    print("\n📝 Creating Configuration:")
    print("="*70)

    env_content = """# LangChain Local Configuration
# No API keys needed for local models!

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama2

# If you want to use OpenAI later, add:
# OPENAI_API_KEY=your-api-key-here
"""

    with open("/Users/ashutosh/Cloude/langchain-hello-world/.env", "w") as f:
        f.write(env_content)

    print("✅ Created .env file with local configuration")

def main():
    print_banner()

    print("\n🎯 This setup will help you run LangChain locally without API keys!")
    print("We'll use Ollama - a free tool to run LLMs on your computer.\n")

    # Check Ollama installation
    if check_ollama_installed():
        print("\n✅ Great! Ollama is already installed.\n")

        # Check if Ollama is running
        test_ollama_connection()

        # List available models
        has_models = list_available_models()

        if not has_models:
            # Offer to download models
            pull_ollama_models()

    else:
        print("\n❌ Ollama is not installed.\n")
        install_ollama_instructions()
        print("\n💡 After installing Ollama, run this script again!")
        return

    # Create .env file
    create_env_file()

    print("\n" + "="*70)
    print("🎉 Setup Complete!")
    print("="*70)
    print("\n📚 Next Steps:")
    print("1. Make sure Ollama is running (ollama serve)")
    print("2. Install Python requirements: pip install -r requirements-local.txt")
    print("3. Run examples: python run_all_local.py")
    print("\n✨ You can now use LangChain without any API keys!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        # Install requests if not available
        import requests
    except ImportError:
        print("Installing requests package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    main()
