from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

# Initialize converter with OCR disabled for faster processing
pipeline_options = PdfFormatOption(
    do_ocr=False,  # Disable OCR for faster processing
)

converter = DocumentConverter(
    format_options={
        "pdf": pipeline_options
    }
)

print("Loading PDF...")
# Load PDF
result = converter.convert("sample.pdf")

# Get document
doc = result.document

# Print document info
print(f"Document loaded successfully!")
print(f"Document title: {doc.name if hasattr(doc, 'name') else 'Unknown'}")

# Get text content
full_text = doc.export_to_markdown()
print(f"\nDocument preview (first 500 chars):")
print(full_text[:500] + "..." if len(full_text) > 500 else full_text)

# Chunk it
print("\n\nChunking document...")
chunker = HybridChunker()
chunks = list(chunker.chunk(doc))

print(f"\nTotal chunks created: {len(chunks)}")

# Preview chunks
print("\n--- First 3 chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i + 1}:")
    print(f"Text: {chunk.text[:200]}..." if len(chunk.text) > 200 else f"Text: {chunk.text}")
    print(f"Page: {chunk.meta.get('page', 'Unknown') if hasattr(chunk, 'meta') else 'No metadata'}")
    print("-" * 50)