import streamlit as st
import os
import json
from datetime import datetime

# Set environment to avoid issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

st.set_page_config(
    page_title="🔬 Visual Grounding RAG",
    page_icon="🧠",
    layout="wide"
)

# Header
st.title("🔬 Visual Grounding RAG - Medical Papers")
st.markdown("See exactly WHERE in PDFs your answers come from!")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Input")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # Model selection
    model = st.selectbox(
        "Model",
        ["deepseek-coder", "mistral", "llama3"],
        help="DeepSeek is best for medical content"
    )
    
    # Query type
    query_type = st.selectbox(
        "Extraction Type",
        ["PICOTT", "Risk of Bias", "Outcomes", "Custom"]
    )
    
    # Custom query
    if query_type == "Custom":
        custom_query = st.text_area("Your question:")
    
    # Process button
    if st.button("🚀 Process", type="primary"):
        if uploaded_file:
            with st.spinner("Processing..."):
                try:
                    # Import our modules
                    from core.pdf_processor import PDFProcessor
                    from core.vector_store import VectorStore
                    from medical.prompts import MedicalPrompts
                    
                    # Initialize
                    processor = PDFProcessor()
                    vector_store = VectorStore()
                    prompts = MedicalPrompts()
                    
                    # Process PDF
                    content = uploaded_file.read()
                    doc_result = processor.process(content)
                    
                    # Get query
                    if query_type == "PICOTT":
                        query = prompts.get_picott_prompt()
                    elif query_type == "Risk of Bias":
                        query = prompts.get_bias_prompt()
                    elif query_type == "Outcomes":
                        query = prompts.get_outcomes_prompt()
                    else:
                        query = custom_query
                    
                    # Search
                    chunks = doc_result['chunks']
                    relevant = vector_store.search(query, chunks, k=3)
                    
                    # Simple answer (without LLM for now)
                    st.session_state['processed'] = True
                    st.session_state['chunks'] = relevant
                    st.session_state['query'] = query
                    st.session_state['filename'] = uploaded_file.name
                    
                    st.success("✅ Processing complete!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload a PDF first")

with col2:
    st.header("📊 Results")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        st.subheader("📍 Visual Grounding Results")
        
        # Show relevant chunks
        st.write(f"**Query**: {st.session_state['query'][:100]}...")
        st.write(f"**Found {len(st.session_state['chunks'])} relevant sections**")
        
        # Display each chunk with location
        for i, chunk in enumerate(st.session_state['chunks']):
            with st.expander(f"Result {i+1}"):
                # Show text
                st.write("**Text:**")
                st.write(chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text)
                
                # Show location if available
                if hasattr(chunk, 'meta') and chunk.meta:
                    try:
                        doc_item = chunk.meta.doc_items[0]
                        prov = doc_item.prov[0]
                        page = prov.page_no
                        bbox = prov.bbox
                        
                        st.write("**Location:**")
                        st.write(f"📄 Page {page}")
                        st.write(f"📍 Position: x={bbox.l:.0f}, y={bbox.t:.0f}")
                        st.write(f"📐 Size: {bbox.r-bbox.l:.0f} × {bbox.t-bbox.b:.0f}")
                        
                        # Visual representation
                        st.code(f"""
╔══════════════════════╗
║   Page {page}         ║
║   ┌─────────────┐    ║
║   │ Found here! │    ║
║   │ x:{bbox.l:.0f} y:{bbox.t:.0f}  │    ║
║   └─────────────┘    ║
╚══════════════════════╝
                        """)
                    except:
                        st.write("Location data available in chunk metadata")
        
        # Export section
        st.subheader("📥 Export")
        
        # Create export data
        export_data = {
            "filename": st.session_state['filename'],
            "query": st.session_state['query'],
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for chunk in st.session_state['chunks']:
            chunk_data = {
                "text": chunk.text[:500],
                "page": "Unknown",
                "position": "Unknown"
            }
            
            if hasattr(chunk, 'meta') and chunk.meta:
                try:
                    page = chunk.meta.doc_items[0].prov[0].page_no
                    bbox = chunk.meta.doc_items[0].prov[0].bbox
                    chunk_data['page'] = page
                    chunk_data['position'] = f"x={bbox.l:.0f}, y={bbox.t:.0f}"
                except:
                    pass
            
            export_data['results'].append(chunk_data)
        
        # Download button
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"visual_grounding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.info("👈 Upload a PDF and click Process to see results")
        
        # Show example
        with st.expander("Example Output"):
            st.write("When you process a PDF, you'll see:")
            st.write("- Text excerpts that answer your query")
            st.write("- Exact page numbers")
            st.write("- Bounding box coordinates (x, y, width, height)")
            st.write("- Visual representation of where text was found")

# Footer
st.markdown("---")
st.caption("Built with Docling for PDF processing and FAISS for semantic search")