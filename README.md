# Docling Visual Grounding RAG

A Retrieval-Augmented Generation (RAG) system with visual grounding capabilities using IBM's Docling for document processing.

## 🎯 What This Does

This project creates a RAG system that can:
1. Parse PDFs while preserving layout information
2. Answer questions about document content
3. Show exactly where in the document the answer was found (visual grounding)

## 📁 Project Structure

```
docling-rag/
├── sample.pdf              # Your test PDF
├── test_docling.py         # Basic Docling test
├── rag_pipeline.py         # RAG with semantic search
├── rag_visual_grounding.py # RAG + visual bounding boxes
├── quick_rag_demo.py       # Simplified demo version
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

1. **Activate virtual environment:**
   ```bash
   source docling-env/bin/activate
   ```

2. **Test basic PDF processing:**
   ```bash
   python test_docling.py
   ```
   This loads your PDF and shows how it's chunked.

3. **Run RAG pipeline:**
   ```bash
   python rag_pipeline.py
   ```
   This searches for relevant content based on queries.

4. **Visual grounding (with bounding boxes):**
   ```bash
   python rag_visual_grounding.py
   ```
   This shows WHERE in the document the answer was found.

## 📊 How It Works

1. **Document Processing**: Docling converts PDFs preserving structure
2. **Chunking**: HybridChunker splits text into semantic chunks
3. **Embeddings**: Sentence transformers create vector representations
4. **Search**: FAISS finds most relevant chunks for queries
5. **Visual Grounding**: Bounding boxes show answer locations on pages

## 🔧 Customization

- Change the query in any script:
  ```python
  query = "Your question here?"
  ```

- Adjust number of results:
  ```python
  D, I = index.search(query_embedding, k=5)  # Top 5 instead of 3
  ```

## 📝 Notes

- First run may be slow (downloading models)
- The sample PDF appears to be a medical research paper about cerebellar infarction
- Visual grounding requires page image generation (slower but more accurate)

## 🎉 Next Steps

1. Add Ollama for local LLM integration
2. Create a web interface with FastAPI
3. Support multiple document formats
4. Add conversation memory