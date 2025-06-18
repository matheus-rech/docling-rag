from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Load PDF
print("Loading and converting PDF...")
converter = DocumentConverter()
result = converter.convert("sample.pdf")

# Get document
doc = result.document

# Print document info
print(f"Document loaded successfully!")

# Export to text
text_content = doc.export_to_markdown()
print(f"\nExtracted {len(text_content)} characters from the PDF")
print("\nFirst 500 characters of the document:")
print("-" * 50)
print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
print("-" * 50)

# Chunk it
print("\nChunking the document...")
chunker = HybridChunker()
chunks = list(chunker.chunk(doc))

print(f"\nTotal chunks created: {len(chunks)}")

# Preview chunks
print("\n--- Preview of first 5 chunks ---")
for i, chunk in enumerate(chunks[:5]):
    print(f"\nChunk {i + 1}:")
    print(f"Text: {chunk.text[:200]}..." if len(chunk.text) > 200 else f"Text: {chunk.text}")
    if hasattr(chunk, 'meta') and chunk.meta:
        print(f"Metadata: {chunk.meta}")
    print("-" * 50)