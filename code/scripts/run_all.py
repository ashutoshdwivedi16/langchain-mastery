"""
LangChain Hello World - Complete Feature Explorer
Run all examples or select specific features to explore
"""

import sys
import importlib

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print(" " * 15 + "🦜 LANGCHAIN HELLO WORLD 🔗")
    print(" " * 10 + "Exploring All LangChain Features")
    print("="*70)

def print_menu():
    """Display feature menu"""
    print("\n📋 Available Features:")
    print("="*70)
    print("  1. Basic LLM Interaction")
    print("  2. Prompt Templates and Chains")
    print("  3. Memory and Conversation Management")
    print("  4. RAG (Retrieval-Augmented Generation) with Vector Stores")
    print("  5. Agents and Tools")
    print("  6. Output Parsers and Structured Data")
    print("  7. Run ALL Examples")
    print("  0. Exit")
    print("="*70)

def run_feature(feature_number):
    """Run a specific feature example"""
    feature_map = {
        1: ("01_basic_llm", "Basic LLM Interaction"),
        2: ("02_prompts_and_chains", "Prompt Templates and Chains"),
        3: ("03_memory_and_conversation", "Memory and Conversation"),
        4: ("04_rag_vector_stores", "RAG with Vector Stores"),
        5: ("05_agents_and_tools", "Agents and Tools"),
        6: ("06_output_parsers", "Output Parsers"),
    }

    if feature_number not in feature_map:
        print(f"❌ Invalid feature number: {feature_number}")
        return False

    module_name, feature_name = feature_map[feature_number]

    print(f"\n{'='*70}")
    print(f"🚀 Running: {feature_name}")
    print('='*70)

    try:
        # Import and run the module
        module = importlib.import_module(module_name)
        module.main()
        print(f"\n✅ Completed: {feature_name}")
        return True
    except Exception as e:
        print(f"\n❌ Error running {feature_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_features():
    """Run all feature examples"""
    print("\n" + "="*70)
    print("🚀 Running ALL Features")
    print("="*70)

    for i in range(1, 7):
        success = run_feature(i)
        if not success:
            print(f"\n⚠️  Feature {i} failed. Continue? (y/n): ", end="")
            response = input().strip().lower()
            if response != 'y':
                print("Stopping execution.")
                return

        print("\n" + "-"*70)
        if i < 6:
            print("Press Enter to continue to next feature...")
            input()

    print("\n" + "="*70)
    print("✅ All features completed!")
    print("="*70)

def interactive_mode():
    """Run in interactive mode with menu"""
    while True:
        print_menu()
        try:
            choice = input("\n👉 Select a feature (0-7): ").strip()

            if choice == '0':
                print("\n👋 Thanks for exploring LangChain! Happy coding!")
                break
            elif choice == '7':
                run_all_features()
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                run_feature(int(choice))
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 7.")

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

    # Check if environment is set up
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment!")
            print("Please create a .env file with your OpenAI API key.")
            print("\nExample:")
            print("  OPENAI_API_KEY=sk-your-api-key-here")
            print("\nPress Enter to continue anyway or Ctrl+C to exit...")
            input()
    except ImportError:
        print("\n⚠️  Required packages not installed.")
        print("Please run: pip install -r requirements.txt")
        return

    # Check for command-line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            run_all_features()
        elif arg.isdigit() and 1 <= int(arg) <= 6:
            run_feature(int(arg))
        else:
            print(f"❌ Invalid argument: {arg}")
            print("\nUsage:")
            print("  python run_all.py        # Interactive mode")
            print("  python run_all.py all    # Run all features")
            print("  python run_all.py 1      # Run feature 1")
            print("  python run_all.py 2      # Run feature 2")
            print("  ... etc")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
