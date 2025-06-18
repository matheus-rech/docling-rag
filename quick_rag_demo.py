from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("=== Visual Grounding RAG Demo ===\n")

# Load from cached result if available (faster)
import pickle
import os

if os.path.exists("cached_doc.pkl"):
    print("Loading cached document...")
    with open("cached_doc.pkl", "rb") as f:
        doc, chunks = pickle.load(f)
else:
    print("Loading PDF document (this may take a moment)...")
    converter = DocumentConverter()
    result = converter.convert("sample.pdf")
    doc = result.document
    
    print("Chunking document...")
    chunker = HybridChunker()
    chunks = list(chunker.chunk(doc))
    
    # Cache for next time
    with open("cached_doc.pkl", "wb") as f:
        pickle.dump((doc, chunks), f)

texts = [c.text for c in chunks]
print(f"✓ Document has {len(chunks)} chunks")

# Embeddings
print("\nCreating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=False)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("✓ Vector search index ready")

# Query
query = "What is the diagnosis or medical condition?"
print(f"\n🔍 Query: '{query}'")
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=3)

# Show results
print("\n📄 Top 3 relevant chunks:")
print("=" * 60)
for idx, i in enumerate(I[0]):
    chunk = chunks[i]
    print(f"\n🎯 Match {idx + 1} (Relevance: {1/(1+D[0][idx]):.2%}):")
    
    # Extract page info
    if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items'):
        if chunk.meta.doc_items and len(chunk.meta.doc_items) > 0:
            page_no = chunk.meta.doc_items[0].prov[0].page_no
            print(f"📍 Page {page_no}")
    
    print("-" * 60)
    print(texts[i][:250] + "..." if len(texts[i]) > 250 else texts[i])

print("\n✅ Demo complete! Run 'python rag_visual_grounding.py' for visual bounding boxes.")