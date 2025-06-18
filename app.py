from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import io
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

# Import our modules
from core.pdf_processor import PDFProcessor
from core.llm_client import OllamaClient
from core.vector_store import VectorStore
from medical.prompts import MedicalPrompts
from medical.extractors import PICOTTExtractor

app = FastAPI(title="Medical RAG - Systematic Review Assistant")

# Initialize components
pdf_processor = PDFProcessor()
llm_client = OllamaClient()
vector_store = VectorStore()
medical_prompts = MedicalPrompts()
picott_extractor = PICOTTExtractor()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical RAG - Systematic Review Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; margin: 20px 0; }
            .results { background: #f5f5f5; padding: 20px; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            select { padding: 5px; margin: 10px 0; }
            .confidence { color: #666; font-size: 0.9em; }
            .visual-grounding { border: 2px solid red; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>🔬 Medical RAG - Systematic Review Assistant</h1>
        
        <div class="upload-area">
            <h3>📄 Upload Medical Paper (PDF)</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf" required>
                
                <h3>🤖 Select Model:</h3>
                <select name="model" id="model">
                    <option value="deepseek-coder">DeepSeek (Recommended for Medical)</option>
                    <option value="mistral">Mistral (Faster)</option>
                    <option value="llama3">Llama 3</option>
                </select>
                
                <h3>🔍 Query Type:</h3>
                <select name="query_type" id="query_type">
                    <option value="custom">Custom Question</option>
                    <option value="picott">Extract PICOTT</option>
                    <option value="bias">Risk of Bias Assessment</option>
                    <option value="outcomes">Primary/Secondary Outcomes</option>
                    <option value="adverse">Adverse Events</option>
                </select>
                
                <div id="customQuery" style="margin-top: 10px;">
                    <textarea name="query" placeholder="Enter your question..." style="width: 100%; height: 60px;"></textarea>
                </div>
                
                <br><br>
                <button type="submit">🚀 Process Document</button>
            </form>
        </div>
        
        <div id="results"></div>
        
        <script>
            document.getElementById('query_type').addEventListener('change', function() {
                const customDiv = document.getElementById('customQuery');
                customDiv.style.display = this.value === 'custom' ? 'block' : 'none';
            });
            
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<p>⏳ Processing...</p>';
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    let html = '<div class="results">';
                    html += '<h3>📊 Results</h3>';
                    
                    if (data.structured_data) {
                        html += '<h4>Structured Extraction:</h4>';
                        for (const [key, value] of Object.entries(data.structured_data)) {
                            html += `<p><strong>${key}:</strong> ${value.text} 
                                     <span class="confidence">(confidence: ${(value.confidence * 100).toFixed(1)}%)</span></p>`;
                        }
                    }
                    
                    html += `<h4>Answer:</h4><p>${data.answer}</p>`;
                    
                    if (data.visual_grounding) {
                        html += `<h4>📍 Source Location:</h4>`;
                        html += `<p>Page ${data.visual_grounding.page}, 
                                 Position: ${data.visual_grounding.bbox}</p>`;
                    }
                    
                    html += '<h4>📥 Export Options:</h4>';
                    html += `<button onclick="downloadCSV('${data.session_id}')">Download CSV</button> `;
                    html += `<button onclick="showPRISMA('${data.session_id}')">Show PRISMA</button>`;
                    
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                } catch (error) {
                    resultsDiv.innerHTML = '<p>❌ Error: ' + error.message + '</p>';
                }
            });
            
            async function downloadCSV(sessionId) {
                window.location.href = `/export/csv/${sessionId}`;
            }
            
            async function showPRISMA(sessionId) {
                const response = await fetch(`/export/prisma/${sessionId}`);
                const data = await response.json();
                alert(data.prisma_text);
            }
        </script>
    </body>
    </html>
    """

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    model: str = Form("deepseek-coder"),
    query_type: str = Form("custom"),
    query: Optional[str] = Form(None)
):
    # Save uploaded file
    content = await file.read()
    
    # Process PDF
    doc_result = pdf_processor.process(content)
    
    # Setup query based on type
    if query_type == "picott":
        actual_query = medical_prompts.get_picott_prompt()
    elif query_type == "bias":
        actual_query = medical_prompts.get_bias_prompt()
    elif query_type == "outcomes":
        actual_query = medical_prompts.get_outcomes_prompt()
    elif query_type == "adverse":
        actual_query = medical_prompts.get_adverse_prompt()
    else:
        actual_query = query or "Summarize this document"
    
    # Get relevant chunks
    relevant_chunks = vector_store.search(actual_query, doc_result['chunks'], k=5)
    
    # Generate answer with confidence scoring
    answer_data = llm_client.generate_with_confidence(
        query=actual_query,
        context=relevant_chunks,
        model=model
    )
    
    # Extract structured data if applicable
    structured_data = None
    if query_type == "picott":
        structured_data = picott_extractor.extract(answer_data['answer'])
    
    # Get visual grounding info
    visual_grounding = None
    if relevant_chunks and hasattr(relevant_chunks[0], 'meta'):
        visual_grounding = {
            "page": relevant_chunks[0].meta.doc_items[0].prov[0].page_no,
            "bbox": str(relevant_chunks[0].meta.doc_items[0].prov[0].bbox)
        }
    
    # Save session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data = {
        "filename": file.filename,
        "query": actual_query,
        "answer": answer_data['answer'],
        "confidence": answer_data['confidence'],
        "structured_data": structured_data,
        "visual_grounding": visual_grounding,
        "model": model
    }
    
    # Store session (in production, use a database)
    os.makedirs("sessions", exist_ok=True)
    with open(f"sessions/{session_id}.json", "w") as f:
        json.dump(session_data, f)
    
    return {
        "session_id": session_id,
        "answer": answer_data['answer'],
        "confidence": answer_data['confidence'],
        "structured_data": structured_data,
        "visual_grounding": visual_grounding
    }

@app.get("/export/csv/{session_id}")
async def export_csv(session_id: str):
    with open(f"sessions/{session_id}.json", "r") as f:
        data = json.load(f)
    
    # Create DataFrame
    if data.get('structured_data'):
        df_data = []
        for key, value in data['structured_data'].items():
            df_data.append({
                "Field": key,
                "Extracted Text": value['text'],
                "Confidence": f"{value['confidence']*100:.1f}%"
            })
        df = pd.DataFrame(df_data)
    else:
        df = pd.DataFrame([{
            "Question": data['query'],
            "Answer": data['answer'],
            "Confidence": f"{data['confidence']*100:.1f}%",
            "Source": data['filename']
        }])
    
    # Return as CSV
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(
        io.StringIO(stream.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=extraction_{session_id}.csv"}
    )
    return response

@app.get("/export/prisma/{session_id}")
async def export_prisma(session_id: str):
    with open(f"sessions/{session_id}.json", "r") as f:
        data = json.load(f)
    
    prisma_text = f"""
PRISMA-Style Summary
====================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Source: {data['filename']}
Model: {data['model']}

Research Question:
{data['query']}

Extracted Information:
{data['answer']}

Confidence Score: {data['confidence']*100:.1f}%

Visual Grounding:
{data.get('visual_grounding', 'Not available')}
"""
    
    return {"prisma_text": prisma_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)