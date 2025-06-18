import sys
from pathlib import Path

# Check if PDF exists
pdf_path = Path("sample.pdf")
if not pdf_path.exists():
    print(f"Error: {pdf_path} not found!")
    sys.exit(1)

print(f"Found PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.chunking import HybridChunker
    
    print("\nInitializing Docling converter...")
    
    # Create minimal pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    
    # Initialize converter
    converter = DocumentConverter(
        pipeline_options=pipeline_options,
        input_format=InputFormat.PDF
    )
    
    print("Converting PDF...")
    result = converter.convert(str(pdf_path))
    
    # Get document
    doc = result.document
    
    print(f"\nDocument converted successfully!")
    
    # Try to get some content
    try:
        text = doc.export_to_markdown()
        print(f"Extracted text length: {len(text)} characters")
        print(f"\nFirst 300 characters:")
        print(text[:300] + "..." if len(text) > 300 else text)
    except Exception as e:
        print(f"Could not export to markdown: {e}")
    
    # Try chunking
    print("\n\nAttempting to chunk document...")
    chunker = HybridChunker()
    chunks = list(chunker.chunk(doc))
    print(f"Created {len(chunks)} chunks")
    
    if chunks:
        print(f"\nFirst chunk preview:")
        print(chunks[0].text[:200] + "..." if len(chunks[0].text) > 200 else chunks[0].text)
    
except ImportError as e:
    print(f"\nImport error: {e}")
    print("\nTrying alternative import structure...")
    
    # Try alternative imports
    try:
        from docling import Document
        from docling.chunking import HybridChunker
        print("Alternative imports successful!")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()