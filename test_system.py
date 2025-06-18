#!/usr/bin/env python3
"""
Test script for Visual Grounding RAG system
Tests all components before running the web interface
"""

import os
import sys
import subprocess
import time

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    try:
        import docling
        print("  ✅ Docling")
        import faiss
        print("  ✅ FAISS")
        import sentence_transformers
        print("  ✅ Sentence Transformers")
        import langchain
        print("  ✅ LangChain")
        import fastapi
        print("  ✅ FastAPI")
        import streamlit
        print("  ✅ Streamlit")
        import pandas
        print("  ✅ Pandas")
        import matplotlib
        print("  ✅ Matplotlib")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_ollama():
    """Test Ollama installation"""
    print("\n🤖 Testing Ollama...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Ollama is running")
            if "deepseek" in result.stdout.lower():
                print("  ✅ DeepSeek model found")
            else:
                print("  ⚠️  DeepSeek not found - pulling model...")
                subprocess.run(["ollama", "pull", "deepseek-coder:latest"])
            return True
    except FileNotFoundError:
        print("  ❌ Ollama not found")
        return False

def test_pdf_processing():
    """Test basic PDF processing"""
    print("\n📄 Testing PDF processing...")
    try:
        from core.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        
        # Check if sample.pdf exists
        if os.path.exists("sample.pdf"):
            print("  ✅ Sample PDF found")
            with open("sample.pdf", "rb") as f:
                content = f.read()
            
            # Test processing
            print("  ⏳ Processing PDF (this may take a moment)...")
            result = processor.process(content)
            print(f"  ✅ PDF processed: {result['num_chunks']} chunks created")
            return True
        else:
            print("  ⚠️  sample.pdf not found")
            return False
    except Exception as e:
        print(f"  ❌ PDF processing error: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\n🔍 Testing vector store...")
    try:
        from core.vector_store import VectorStore
        store = VectorStore()
        
        # Test with dummy data
        class DummyChunk:
            def __init__(self, text):
                self.text = text
        
        chunks = [
            DummyChunk("Patient with cerebellar infarction"),
            DummyChunk("Treatment with SDC surgery"),
            DummyChunk("Outcome measures included")
        ]
        
        results = store.search("What is the treatment?", chunks, k=2)
        print(f"  ✅ Vector search working: {len(results)} results found")
        return True
    except Exception as e:
        print(f"  ❌ Vector store error: {e}")
        return False

def run_demo():
    """Run a quick demo"""
    print("\n🎯 Running quick demo...")
    print("=" * 50)
    
    try:
        from core.pdf_processor import PDFProcessor
        from core.vector_store import VectorStore
        from medical.prompts import MedicalPrompts
        
        processor = PDFProcessor()
        vector_store = VectorStore()
        prompts = MedicalPrompts()
        
        if os.path.exists("sample.pdf"):
            print("📄 Processing sample.pdf...")
            with open("sample.pdf", "rb") as f:
                content = f.read()
            
            doc_result = processor.process(content)
            print(f"✅ Created {doc_result['num_chunks']} chunks")
            
            # Search for medical information
            query = "What is the patient diagnosis and treatment?"
            print(f"\n🔍 Query: '{query}'")
            
            chunks = doc_result['chunks']
            relevant = vector_store.search(query, chunks, k=3)
            
            print(f"\n📊 Found {len(relevant)} relevant chunks:")
            for i, chunk in enumerate(relevant[:2]):
                print(f"\nChunk {i+1}:")
                print(f"Text: {chunk.text[:150]}...")
                if hasattr(chunk, 'meta') and chunk.meta:
                    if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                        page = chunk.meta.doc_items[0].prov[0].page_no
                        print(f"📍 Location: Page {page}")
            
            print("\n✅ Demo complete!")
            return True
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def main():
    print("🔬 Visual Grounding RAG - System Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Ollama", test_ollama),
        ("PDF Processing", test_pdf_processing),
        ("Vector Store", test_vector_store)
    ]
    
    all_passed = True
    for name, test_func in tests:
        if not test_func():
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
        
        # Run demo
        run_demo()
        
        print("\n🚀 Ready to start the web interface!")
        print("\nChoose an option:")
        print("1. FastAPI interface (recommended)")
        print("2. Streamlit interface")
        print("3. Batch processing interface")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            print("\n🌐 Starting FastAPI interface...")
            print("📍 Open http://localhost:8000 in your browser")
            os.system("python -m uvicorn app:app --reload")
        elif choice == "2":
            print("\n🌐 Starting Streamlit interface...")
            os.system("streamlit run streamlit_visual_app.py")
        elif choice == "3":
            print("\n🌐 Starting batch processing interface...")
            print("📍 Open http://localhost:8001 in your browser")
            os.system("python -m uvicorn app_batch:app --reload --port 8001")
        else:
            print("\nExiting...")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install Ollama: brew install ollama")
        print("- Pull model: ollama pull deepseek-coder:latest")
        print("- Install deps: pip install -r requirements.txt")

if __name__ == "__main__":
    main()