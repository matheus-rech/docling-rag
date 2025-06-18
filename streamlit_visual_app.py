import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import io
import base64
from datetime import datetime

# Import our enhanced modules
from core.pdf_processor import PDFProcessor
from core.llm_client import OllamaClient
from core.vector_store import VectorStore
from medical.prompts import MedicalPrompts
from medical.extractors import PICOTTExtractor

# Initialize components
@st.cache_resource
def init_components():
    return (
        PDFProcessor(),
        OllamaClient(),
        VectorStore(),
        MedicalPrompts(),
        PICOTTExtractor()
    )

pdf_processor, llm_client, vector_store, medical_prompts, picott_extractor = init_components()

# Streamlit config
st.set_page_config(
    page_title="🔬 Visual Grounding RAG - Medical Papers",
    page_icon="🧠",
    layout="wide"
)

st.title("🔬 Visual Grounding RAG for Medical Papers")
st.markdown("Extract information and see **exactly where** it comes from in the PDF")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    model = st.selectbox(
        "🤖 Model",
        ["deepseek-coder", "mistral", "llama3"],
        help="DeepSeek recommended for medical content"
    )
    
    extraction_mode = st.radio(
        "📋 Extraction Mode",
        ["Single PDF", "Batch Processing"],
        help="Process one PDF with visual grounding or multiple PDFs"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Prompts")
    quick_prompts = {
        "PICOTT": medical_prompts.get_picott_prompt(),
        "Risk of Bias": medical_prompts.get_bias_prompt(),
        "Outcomes": medical_prompts.get_outcomes_prompt(),
        "Adverse Events": medical_prompts.get_adverse_prompt(),
        "Custom": ""
    }
    prompt_choice = st.selectbox("Select prompt template", list(quick_prompts.keys()))

# Main content area
if extraction_mode == "Single PDF":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload & Query")
        uploaded_file = st.file_uploader("Upload medical paper (PDF)", type="pdf")
        
        if prompt_choice == "Custom":
            query = st.text_area("Enter your question:", height=100)
        else:
            query = quick_prompts[prompt_choice]
            st.text_area("Selected prompt:", value=query, height=100, disabled=True)
        
        if uploaded_file and query and st.button("🔍 Extract with Visual Grounding", type="primary"):
            with st.spinner("Processing PDF..."):
                # Process PDF
                content = uploaded_file.read()
                doc_result = pdf_processor.process(content)
                
                # Get relevant chunks
                relevant_chunks = vector_store.search(query, doc_result['chunks'], k=5)
                
                # Generate answer with confidence
                answer_data = llm_client.generate_with_confidence(
                    query=query,
                    context=relevant_chunks,
                    model=model
                )
                
                # Store in session state
                st.session_state['answer'] = answer_data
                st.session_state['chunks'] = relevant_chunks
                st.session_state['doc_result'] = doc_result
                st.session_state['filename'] = uploaded_file.name
                
                # Process with images for visual grounding
                with st.spinner("Generating visual grounding..."):
                    st.session_state['visual_result'] = pdf_processor.process_with_images(content)
    
    with col2:
        st.header("📊 Results")
        
        if 'answer' in st.session_state:
            # Display answer with confidence
            st.subheader("Answer")
            st.write(st.session_state['answer']['answer'])
            
            # Confidence meter
            confidence = st.session_state['answer']['confidence']
            st.metric("Confidence Score", f"{confidence*100:.1f}%")
            
            # Progress bar for confidence
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
            st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
            
            # Visual grounding section
            st.subheader("📍 Visual Grounding")
            
            if st.session_state['chunks']:
                # Get first chunk with location info
                chunk = st.session_state['chunks'][0]
                if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items'):
                    doc_item = chunk.meta.doc_items[0]
                    prov = doc_item.prov[0]
                    page_no = prov.page_no - 1
                    bbox = prov.bbox
                    
                    st.info(f"Found on page {page_no + 1}")
                    
                    # Try to display page with bounding box
                    if 'visual_result' in st.session_state:
                        visual_result = st.session_state['visual_result']
                        if hasattr(visual_result, 'pages') and len(visual_result.pages) > page_no:
                            page = visual_result.pages[page_no]
                            if hasattr(page, 'image') and page.image:
                                # Create figure
                                fig, ax = plt.subplots(figsize=(10, 12))
                                ax.imshow(page.image)
                                
                                # Convert coordinates (Docling uses bottom-left origin)
                                img_height = page.image.height
                                x = bbox.l
                                y = img_height - bbox.t
                                width = bbox.r - bbox.l
                                height = bbox.t - bbox.b
                                
                                # Draw bounding box
                                rect = patches.Rectangle(
                                    (x, y), width, height,
                                    linewidth=3, edgecolor='red', facecolor='none'
                                )
                                ax.add_patch(rect)
                                
                                ax.axis('off')
                                ax.set_title(f"Visual Grounding - Page {page_no + 1}", fontsize=16)
                                
                                # Display in Streamlit
                                st.pyplot(fig)
                                plt.close()
            
            # Export options
            st.subheader("📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                export_data = {
                    "Filename": [st.session_state['filename']],
                    "Query": [query],
                    "Answer": [st.session_state['answer']['answer']],
                    "Confidence": [f"{confidence*100:.1f}%"],
                    "Page": [page_no + 1 if 'page_no' in locals() else "N/A"]
                }
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # PRISMA export
                prisma_text = f"""# Extraction Summary (PRISMA Format)
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source: {st.session_state['filename']}
Model: {model}

## Research Question
{query}

## Extracted Information
{st.session_state['answer']['answer']}

## Quality Assessment
Confidence Score: {confidence*100:.1f}%

## Visual Grounding
Page: {page_no + 1 if 'page_no' in locals() else 'N/A'}
Location: {f"x={bbox.l:.0f}, y={bbox.t:.0f}" if 'bbox' in locals() else 'N/A'}
"""
                
                st.download_button(
                    label="📄 Download PRISMA",
                    data=prisma_text,
                    file_name=f"prisma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

else:  # Batch Processing Mode
    st.header("📚 Batch Processing")
    
    uploaded_files = st.file_uploader(
        "Upload multiple PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded")
        
        if prompt_choice == "Custom":
            batch_query = st.text_area("Enter your question for all PDFs:", height=100)
        else:
            batch_query = quick_prompts[prompt_choice]
            st.text_area("Selected prompt:", value=batch_query, height=100, disabled=True)
        
        if batch_query and st.button("🚀 Process All PDFs", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                try:
                    content = file.read()
                    doc_result = pdf_processor.process(content)
                    relevant_chunks = vector_store.search(batch_query, doc_result['chunks'], k=3)
                    
                    answer_data = llm_client.generate_with_confidence(
                        query=batch_query,
                        context=relevant_chunks,
                        model=model
                    )
                    
                    results.append({
                        "PDF": file.name,
                        "Answer": answer_data['answer'][:200] + "...",
                        "Confidence": f"{answer_data['confidence']*100:.1f}%",
                        "Full Answer": answer_data['answer']
                    })
                except Exception as e:
                    results.append({
                        "PDF": file.name,
                        "Answer": f"Error: {str(e)}",
                        "Confidence": "0%",
                        "Full Answer": str(e)
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            
            # Display results table
            df = pd.DataFrame(results)
            st.dataframe(df[["PDF", "Answer", "Confidence"]], use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📊 Download Full Results (CSV)",
                    csv,
                    "batch_results.csv",
                    "text/csv"
                )
            
            with col2:
                # Create detailed markdown report
                report = "# Batch Extraction Report\n\n"
                for _, row in df.iterrows():
                    report += f"## {row['PDF']}\n\n"
                    report += f"**Confidence:** {row['Confidence']}\n\n"
                    report += f"**Answer:**\n{row['Full Answer']}\n\n---\n\n"
                
                st.download_button(
                    "📄 Download Report (Markdown)",
                    report,
                    "batch_report.md",
                    "text/markdown"
                )

# Footer
st.markdown("---")
st.markdown("🔬 **Visual Grounding RAG** - See exactly where information comes from in medical papers")