#!/usr/bin/env python3
"""
Medical RAG - Systematic Review Assistant
Start the application
"""

import os
import sys
import subprocess

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            models = result.stdout
            if "deepseek" not in models:
                print("⚠️  DeepSeek model not found. Installing...")
                subprocess.run(["ollama", "pull", "deepseek-coder:latest"])
            return True
    except FileNotFoundError:
        print("❌ Ollama not found. Please install it first:")
        print("   brew install ollama")
        print("   ollama pull deepseek-coder:latest")
        return False
    return False

def check_dependencies():
    """Check if all Python dependencies are installed"""
    try:
        import fastapi
        import langchain
        import docling
        import sentence_transformers
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    print("🔬 Medical RAG - Systematic Review Assistant")
    print("=" * 50)
    
    # Check Ollama
    if not check_ollama():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting server...")
    print("📍 Open http://localhost:8000 in your browser")
    print("\nPress Ctrl+C to stop\n")
    
    # Start the app
    os.system("python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()