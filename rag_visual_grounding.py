from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

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

# Visual Grounding - show bounding box from best chunk
print("\n\nPerforming visual grounding...")
best_chunk = chunks[I[0][0]]

# Get bounding box info from chunk metadata
if hasattr(best_chunk, 'meta') and best_chunk.meta and hasattr(best_chunk.meta, 'doc_items'):
    # Extract page number and bounding box from first doc_item
    doc_items = best_chunk.meta.doc_items
    if doc_items and len(doc_items) > 0:
        first_item = doc_items[0]
        if hasattr(first_item, 'prov') and first_item.prov:
            prov = first_item.prov[0]
            page_no = prov.page_no - 1  # 0-indexed
            bbox = prov.bbox
            
            print(f"Found answer on page {page_no + 1}")
            print(f"Bounding box: l={bbox.l:.1f}, t={bbox.t:.1f}, r={bbox.r:.1f}, b={bbox.b:.1f}")
            
            # Try to get page image
            try:
                # Generate page image if not available
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                
                pipeline_options = PdfPipelineOptions()
                pipeline_options.generate_page_images = True
                pipeline_options.images_scale = 2.0  # Higher resolution
                
                converter_with_images = DocumentConverter(
                    format_options={
                        "pdf": pipeline_options
                    }
                )
                
                print("Generating page image...")
                result_with_images = converter_with_images.convert("sample.pdf")
                
                # Get the page image
                if hasattr(result_with_images, 'pages') and len(result_with_images.pages) > page_no:
                    page = result_with_images.pages[page_no]
                    if hasattr(page, 'image') and page.image:
                        img = page.image
                        
                        # Create visualization
                        plt.figure(figsize=(12, 8))
                        plt.imshow(img)
                        
                        # Draw bounding box
                        # Note: Docling uses bottom-left origin, matplotlib uses top-left
                        # Convert coordinates
                        img_height = img.height
                        x = bbox.l
                        y = img_height - bbox.t  # Flip Y coordinate
                        width = bbox.r - bbox.l
                        height = bbox.t - bbox.b
                        
                        rect = plt.Rectangle((x, y), width, height, 
                                           fill=False, color='red', linewidth=3)
                        plt.gca().add_patch(rect)
                        
                        plt.axis('off')
                        plt.title(f"Visual Grounding Result - Page {page_no + 1}\nQuery: '{query}'", 
                                fontsize=14, pad=20)
                        plt.tight_layout()
                        
                        # Save the figure
                        plt.savefig('visual_grounding_result.png', dpi=150, bbox_inches='tight')
                        print("\nVisual grounding result saved as 'visual_grounding_result.png'")
                        
                        plt.show()
                    else:
                        print("Could not get page image")
                else:
                    print("Page not found in document")
                    
            except Exception as e:
                print(f"Error creating visualization: {e}")
else:
    print("No bounding box information available for visual grounding")