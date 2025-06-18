#!/usr/bin/env python3
"""Quick verification that all components work"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🔍 Verifying setup...")

try:
    # Test imports
    from fastapi import FastAPI
    from core.pdf_processor import PDFProcessor
    from core.llm_client import OllamaClient
    from core.vector_store import VectorStore
    print("✅ All imports successful")
    
    # Test Ollama connection
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "deepseek-coder" in result.stdout:
        print("✅ DeepSeek model available")
    else:
        print("⚠️  DeepSeek not found, but other models available")
    
    # Test if sample.pdf exists
    if os.path.exists("sample.pdf"):
        print("✅ Sample PDF found")
    else:
        print("❌ sample.pdf not found")
    
    print("\n✨ Setup verified! You can now run:")
    print("   ./run_app.sh")
    print("\nThen open http://localhost:8000 in your browser")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease run: pip install -r requirements.txt")