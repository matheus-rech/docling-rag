from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
import pickle
import os
import hashlib

class PDFProcessor:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()
    
    def _get_cache_path(self, content):
        """Generate cache filename from content hash"""
        content_hash = hashlib.md5(content).hexdigest()
        return os.path.join(self.cache_dir, f"doc_{content_hash}.pkl")
    
    def process(self, pdf_content):
        """Process PDF with caching"""
        cache_path = self._get_cache_path(pdf_content)
        
        # Check cache
        if os.path.exists(cache_path):
            print("Loading from cache...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Save content to temp file
        temp_path = os.path.join(self.cache_dir, "temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_content)
        
        # Convert document
        result = self.converter.convert(temp_path)
        doc = result.document
        
        # Extract text and chunks
        chunks = list(self.chunker.chunk(doc))
        text_content = doc.export_to_markdown()
        
        # Prepare result
        doc_data = {
            "chunks": chunks,
            "text": text_content,
            "num_chunks": len(chunks),
            "pages": len(result.pages) if hasattr(result, 'pages') else 0
        }
        
        # Cache result
        with open(cache_path, "wb") as f:
            pickle.dump(doc_data, f)
        
        # Cleanup
        os.remove(temp_path)
        
        return doc_data
    
    def process_with_images(self, pdf_content):
        """Process PDF with page images for visual grounding"""
        # Create converter with image generation
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.images_scale = 2.0
        
        converter_with_images = DocumentConverter(
            format_options={"pdf": pipeline_options}
        )
        
        # Save content to temp file
        temp_path = os.path.join(self.cache_dir, "temp_img.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_content)
        
        # Convert
        result = converter_with_images.convert(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return result