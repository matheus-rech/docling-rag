#!/usr/bin/env python3
"""
Quick test without multiprocessing issues
"""

import os
import sys

# Set environment variable to avoid multiprocessing issues on Mac
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

def test_basic_setup():
    print("🔬 Quick System Check")
    print("=" * 50)
    
    # Test imports
    print("\n✅ Testing core imports...")
    try:
        import docling
        import faiss
        import sentence_transformers
        import pandas
        import matplotlib
        print("  All core libraries loaded successfully!")
    except ImportError as e:
        print(f"  ❌ Missing: {e}")
        return False
    
    # Test Ollama
    print("\n✅ Checking Ollama...")
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    print("  Available models:")
    for line in result.stdout.split('\n'):
        if line.strip() and not line.startswith('NAME'):
            print(f"    - {line.split()[0]}")
    
    return True

def test_simple_rag():
    print("\n🧪 Testing Simple RAG Pipeline...")
    
    try:
        # Import with minimal setup
        from docling.document_converter import DocumentConverter
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        
        print("  ✅ Imports successful")
        
        # Check if sample.pdf exists
        if not os.path.exists("sample.pdf"):
            print("  ❌ sample.pdf not found")
            return False
        
        print("  📄 Found sample.pdf")
        
        # Simple embeddings test
        print("  🔍 Testing embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        test_texts = ["Patient diagnosis", "Treatment protocol", "Clinical outcomes"]
        embeddings = model.encode(test_texts)
        print(f"  ✅ Created {len(embeddings)} embeddings")
        
        # Test FAISS
        print("  🔍 Testing vector search...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        query = model.encode(["What is the diagnosis?"])
        D, I = index.search(query, k=2)
        print(f"  ✅ Search returned {len(I[0])} results")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def suggest_next_steps():
    print("\n📋 Next Steps:")
    print("=" * 50)
    
    print("\n1. For FastAPI interface (without multiprocessing issues):")
    print("   export TOKENIZERS_PARALLELISM=false")
    print("   python -m uvicorn app:app --reload --workers 1")
    
    print("\n2. For Streamlit interface:")
    print("   streamlit run streamlit_visual_app.py")
    
    print("\n3. For basic PDF test:")
    print("   python test_docling.py")
    
    print("\n4. To use available models:")
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    models = [line.split()[0] for line in result.stdout.split('\n') if line.strip() and not line.startswith('NAME')]
    
    if models:
        print(f"   Available: {', '.join(models)}")
        if 'deepseek-coder' in models:
            print("   ✅ DeepSeek is ready!")
        elif 'mistral' in models:
            print("   💡 Use Mistral instead of DeepSeek")

def main():
    # Run tests
    if test_basic_setup():
        test_simple_rag()
    
    # Provide guidance
    suggest_next_steps()
    
    print("\n✨ Ready to start! Choose an option above.")

if __name__ == "__main__":
    # Ensure we're using spawn method to avoid segfault
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()