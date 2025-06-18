from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load document and chunk
print("Loading PDF document...")
converter = DocumentConverter()
result = converter.convert("sample.pdf")
doc = result.document

print("Chunking document...")
chunker = HybridChunker()
chunks = list(chunker.chunk(doc))
texts = [c.text for c in chunks]

print(f"Created {len(chunks)} chunks from the document")

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Create FAISS index
print("Creating FAISS vector index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Simulate a user query
query = "What is the patient's diagnosis?"
print(f"\nSearching for: '{query}'")
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=3)

# Print top matches
print("\nTop 3 relevant chunks:")
print("=" * 50)
for idx, i in enumerate(I[0]):
    print(f"\nMatch {idx + 1} (Distance: {D[0][idx]:.4f}):")
    print("-" * 50)
    print(f"{texts[i][:300]}..." if len(texts[i]) > 300 else texts[i])
    print("-" * 50)