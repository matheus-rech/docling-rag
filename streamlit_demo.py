import streamlit as st
import os
import logging

# Set environment to avoid issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Setup logging
logger = logging.getLogger(__name__)
# You might want to configure the logger further (level, handler, formatter)
# For a simple Streamlit app, basicConfig might be sufficient if not configured elsewhere.
# logging.basicConfig(level=logging.INFO) # Example: set logging level for the app

st.set_page_config(
    page_title="🔬 Visual Grounding RAG Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🔬 Visual Grounding RAG for Medical Papers")
st.markdown("Extract information from PDFs with exact location visualization")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app demonstrates:
    - 📄 PDF processing with Docling
    - 🔍 Semantic search with FAISS
    - 🤖 LLM integration (DeepSeek/Mistral)
    - 📍 Visual grounding (bounding boxes)
    - 📊 Confidence scoring
    """)
    
    st.header("🎯 Quick Start")
    st.markdown("""
    1. Upload a PDF
    2. Select extraction type
    3. Click Process
    4. See results with visual grounding
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload & Configure")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a medical/scientific paper"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    extraction_type = st.selectbox(
        "Select extraction type",
        ["PICOTT Extraction", "Risk of Bias", "Outcomes", "Custom Query"],
        help="Choose what to extract from the PDF"
    )
    
    if extraction_type == "Custom Query":
        custom_query = st.text_area(
            "Enter your question:",
            placeholder="What is the patient population?",
            height=100
        )
    
    model = st.selectbox(
        "Select model",
        ["deepseek-coder", "mistral", "llama3"],
        help="DeepSeek is recommended for medical content"
    )
    
    if st.button("🚀 Process Document", type="primary", disabled=not uploaded_file):
        with st.spinner("Processing... This may take a moment"):
            # Import components
            try:
                from core.pdf_processor import PDFProcessor
                from core.llm_client import OllamaClient
                from core.vector_store import VectorStore
                from medical.prompts import MedicalPrompts
                
                # Initialize
                processor = PDFProcessor()
                llm = OllamaClient()
                vector_store = VectorStore()
                prompts = MedicalPrompts()
                
                # Process PDF
                content = uploaded_file.read()
                doc_result = processor.process(content)

                if not doc_result or not doc_result.get('chunks'):
                    st.error("Failed to process the PDF or extract any content chunks. Please try another PDF or check the document format.")
                    # Optional: log this specific failure point
                    logger.error("PDF processing failed or yielded no chunks for file: %s", uploaded_file.name if uploaded_file else "Unknown")
                    return # Stop further processing
                
                chunks_from_pdf = doc_result['chunks'] # Renamed to avoid conflict if 'chunks' is used later for 'relevant'

                # Get query
                if extraction_type == "PICOTT Extraction":
                    query = prompts.get_picott_prompt()
                elif extraction_type == "Risk of Bias":
                    query = prompts.get_bias_prompt()
                elif extraction_type == "Outcomes":
                    query = prompts.get_outcomes_prompt()
                else:
                    query = custom_query
                
                # Search and generate answer
                # Ensure 'query' is defined before this block, which it is based on context
                relevant_chunks = vector_store.search(query, chunks_from_pdf, k=3)

                if not relevant_chunks:
                    st.warning("Could not retrieve relevant segments from the document for your query. This could be due to the query itself, or the document content. The LLM will try to answer based on a general understanding if possible.")
                    # Log this situation
                    logger.warning("Vector search returned no relevant chunks for query: '%s' in file: %s", query, uploaded_file.name if uploaded_file else "Unknown")

                answer_data = llm.generate_with_confidence(query, relevant_chunks, model)
                
                # Store in session state
                st.session_state['results'] = {
                    'answer': answer_data['answer'],
                    'confidence': answer_data['confidence'],
                    'chunks': relevant_chunks, # Store the actual relevant chunks used for generation
                    'filename': uploaded_file.name
                }
                
                st.success("✅ Processing complete!")
                
            except Exception as e:
                logger.error(f"Streamlit processing error: {str(e)}", exc_info=True) # exc_info=True logs traceback
                st.error(f"An error occurred during processing: {str(e)}")

with col2:
    st.header("📊 Results")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Display answer
        st.subheader("Extracted Information")
        st.write(results['answer'])
        
        # Confidence score
        confidence = results['confidence']
        st.metric(
            "Confidence Score",
            f"{confidence*100:.1f}%",
            delta=None,
            delta_color="normal"
        )
        
        # Progress bar
        st.progress(confidence)
        
        # Visual grounding info
        st.subheader("📍 Source Location")
        if results['chunks']:
            chunk = results['chunks'][0]
            if hasattr(chunk, 'meta') and chunk.meta:
                try:
                    page = chunk.meta.doc_items[0].prov[0].page_no
                    bbox = chunk.meta.doc_items[0].prov[0].bbox
                    st.info(f"Found on page {page}")
                    st.code(f"Bounding box: x={bbox.l:.0f}, y={bbox.t:.0f}, width={bbox.r-bbox.l:.0f}, height={bbox.t-bbox.b:.0f}")
                except:
                    st.info("Location information available in chunks")
        
        # Export options
        st.subheader("📥 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download CSV"):
                import pandas as pd
                df = pd.DataFrame([{
                    'File': results['filename'],
                    'Answer': results['answer'],
                    'Confidence': f"{confidence*100:.1f}%"
                }])
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download",
                    csv,
                    "results.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("📄 Show PRISMA"):
                prisma = f"""
# PRISMA Summary
**File**: {results['filename']}
**Confidence**: {confidence*100:.1f}%

## Extracted Information
{results['answer']}
"""
                st.markdown(prisma)
    else:
        st.info("👆 Upload a PDF and click Process to see results")

# Footer
st.markdown("---")
st.caption("🔬 Visual Grounding RAG - Built with Docling, FAISS, and Ollama")