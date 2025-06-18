from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import pandas as pd
import io
import json
from datetime import datetime
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.pdf_processor import PDFProcessor
from core.llm_client import OllamaClient
from core.vector_store import VectorStore
from medical.prompts import MedicalPrompts
from medical.extractors import PICOTTExtractor

app = FastAPI(title="Medical RAG - Batch Systematic Review")

# Initialize components
pdf_processor = PDFProcessor()
llm_client = OllamaClient()
vector_store = VectorStore()
medical_prompts = MedicalPrompts()
picott_extractor = PICOTTExtractor()

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical RAG - Batch Systematic Review</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; margin: 20px 0; background: #f9f9f9; }
            .results-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            .results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .results-table th { background-color: #4CAF50; color: white; }
            .results-table tr:nth-child(even) { background-color: #f2f2f2; }
            .confidence { color: #666; font-size: 0.9em; }
            .high-confidence { color: green; }
            .medium-confidence { color: orange; }
            .low-confidence { color: red; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; margin: 5px; }
            .progress { width: 100%; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0; }
            .progress-bar { height: 30px; background-color: #4CAF50; text-align: center; line-height: 30px; color: white; border-radius: 5px; }
            .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
            .study-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .study-card h4 { margin-top: 0; color: #333; }
        </style>
    </head>
    <body>
        <h1>🔬 Medical RAG - Batch Systematic Review</h1>
        <p>Process multiple clinical trial PDFs for systematic review data extraction</p>
        
        <div class="upload-area">
            <h3>📄 Upload Multiple PDFs</h3>
            <form id="batchForm" enctype="multipart/form-data">
                <input type="file" name="files" accept=".pdf" multiple required>
                <p style="color: #666;">Select all PDFs you want to process</p>
                
                <h3>🤖 Select Model:</h3>
                <select name="model" id="model">
                    <option value="deepseek-coder">DeepSeek (Best for Medical)</option>
                    <option value="mistral">Mistral (Faster)</option>
                </select>
                
                <h3>🔍 Extraction Type:</h3>
                <select name="extraction_type" id="extraction_type">
                    <option value="picott">PICOTT Extraction</option>
                    <option value="bias">Risk of Bias Assessment</option>
                    <option value="outcomes">Outcomes & Results</option>
                    <option value="full_review">Full Systematic Review</option>
                    <option value="custom">Custom Query</option>
                </select>
                
                <div id="customQuery" style="display: none; margin-top: 10px;">
                    <textarea name="custom_query" placeholder="Enter your custom query..." style="width: 100%; height: 60px;"></textarea>
                </div>
                
                <br><br>
                <button type="submit">🚀 Process All PDFs</button>
            </form>
        </div>
        
        <div id="progress" style="display: none;">
            <h3>📊 Processing Progress</h3>
            <div class="progress">
                <div class="progress-bar" id="progressBar" style="width: 0%">0%</div>
            </div>
            <p id="currentFile"></p>
        </div>
        
        <div id="results"></div>
        
        <script>
            document.getElementById('extraction_type').addEventListener('change', function() {
                const customDiv = document.getElementById('customQuery');
                customDiv.style.display = this.value === 'custom' ? 'block' : 'none';
            });
            
            document.getElementById('batchForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const files = formData.getAll('files');
                const totalFiles = files.length;
                
                document.getElementById('progress').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                
                const results = [];
                let processed = 0;
                
                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    document.getElementById('currentFile').textContent = `Processing: ${file.name}`;
                    
                    const fileFormData = new FormData();
                    fileFormData.append('file', file);
                    fileFormData.append('model', formData.get('model'));
                    fileFormData.append('extraction_type', formData.get('extraction_type'));
                    fileFormData.append('custom_query', formData.get('custom_query') || '');
                    
                    try {
                        const response = await fetch('/process_single', {
                            method: 'POST',
                            body: fileFormData
                        });
                        const data = await response.json();
                        results.push({...data, filename: file.name});
                    } catch (error) {
                        results.push({
                            filename: file.name,
                            error: error.message
                        });
                    }
                    
                    processed++;
                    const progress = Math.round((processed / totalFiles) * 100);
                    document.getElementById('progressBar').style.width = progress + '%';
                    document.getElementById('progressBar').textContent = progress + '%';
                }
                
                displayResults(results);
            });
            
            function displayResults(results) {
                let html = '<h2>📋 Extraction Results</h2>';
                
                // Summary statistics
                const successful = results.filter(r => !r.error).length;
                html += `<p>✅ Successfully processed: ${successful}/${results.length} PDFs</p>`;
                
                // Results table
                html += '<table class="results-table">';
                html += '<thead><tr><th>PDF</th><th>Extracted Data</th><th>Confidence</th><th>Actions</th></tr></thead>';
                html += '<tbody>';
                
                results.forEach((result, idx) => {
                    if (result.error) {
                        html += `<tr><td>${result.filename}</td><td colspan="3" style="color: red;">Error: ${result.error}</td></tr>`;
                    } else {
                        const avgConfidence = result.average_confidence || 0;
                        const confClass = avgConfidence > 0.7 ? 'high-confidence' : avgConfidence > 0.5 ? 'medium-confidence' : 'low-confidence';
                        
                        html += `<tr>`;
                        html += `<td>${result.filename}</td>`;
                        html += `<td>${formatExtractedData(result)}</td>`;
                        html += `<td class="${confClass}">${(avgConfidence * 100).toFixed(1)}%</td>`;
                        html += `<td><button onclick="viewDetails(${idx})">View Details</button></td>`;
                        html += `</tr>`;
                    }
                });
                
                html += '</tbody></table>';
                
                // Export buttons
                html += '<div style="margin-top: 20px;">';
                html += '<button onclick="exportCSV()">📥 Export to CSV</button>';
                html += '<button onclick="exportPRISMA()">📄 Export PRISMA Summary</button>';
                html += '<button onclick="showComparison()">📊 Compare Studies</button>';
                html += '</div>';
                
                // Comparison view
                html += '<div id="comparisonView" style="display: none;">';
                html += '<h3>📊 Study Comparison</h3>';
                html += '<div class="comparison-grid">';
                
                results.filter(r => !r.error).forEach(result => {
                    html += `<div class="study-card">`;
                    html += `<h4>${result.filename}</h4>`;
                    if (result.structured_data) {
                        Object.entries(result.structured_data).forEach(([key, value]) => {
                            html += `<p><strong>${key}:</strong> ${value.text}</p>`;
                        });
                    }
                    html += `</div>`;
                });
                
                html += '</div></div>';
                
                document.getElementById('results').innerHTML = html;
                
                // Store results globally for export
                window.batchResults = results;
            }
            
            function formatExtractedData(result) {
                if (result.structured_data) {
                    return Object.entries(result.structured_data)
                        .map(([key, value]) => `<strong>${key}:</strong> ${value.text.substring(0, 50)}...`)
                        .join('<br>');
                }
                return result.answer ? result.answer.substring(0, 100) + '...' : 'No data extracted';
            }
            
            function viewDetails(idx) {
                const result = window.batchResults[idx];
                alert(JSON.stringify(result, null, 2));
            }
            
            async function exportCSV() {
                const response = await fetch('/export/batch_csv', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(window.batchResults)
                });
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'systematic_review_data.csv';
                a.click();
            }
            
            async function exportPRISMA() {
                const response = await fetch('/export/batch_prisma', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(window.batchResults)
                });
                const data = await response.json();
                const blob = new Blob([data.prisma_text], {type: 'text/markdown'});
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'systematic_review_prisma.md';
                a.click();
            }
            
            function showComparison() {
                const compView = document.getElementById('comparisonView');
                compView.style.display = compView.style.display === 'none' ? 'block' : 'none';
            }
        </script>
    </body>
    </html>
    """

@app.post("/process_single")
async def process_single_pdf(
    file: UploadFile = File(...),
    model: str = Form("deepseek-coder"),
    extraction_type: str = Form("picott"),
    custom_query: str = Form("")
):
    """Process a single PDF"""
    content = await file.read()
    
    # Process PDF
    doc_result = pdf_processor.process(content)
    
    # Determine query
    if extraction_type == "picott":
        query = medical_prompts.get_picott_prompt()
    elif extraction_type == "bias":
        query = medical_prompts.get_bias_prompt()
    elif extraction_type == "outcomes":
        query = medical_prompts.get_outcomes_prompt()
    elif extraction_type == "full_review":
        query = """Extract all key information for systematic review:
        1. PICOTT elements
        2. Risk of bias assessment
        3. Primary and secondary outcomes with results
        4. Adverse events
        5. Study limitations"""
    else:
        query = custom_query
    
    # Get relevant chunks
    relevant_chunks = vector_store.search(query, doc_result['chunks'], k=5)
    
    # Generate answer
    answer_data = llm_client.generate_with_confidence(
        query=query,
        context=relevant_chunks,
        model=model
    )
    
    # Extract structured data
    structured_data = None
    if extraction_type == "picott":
        structured_data = picott_extractor.extract(answer_data['answer'])
    elif extraction_type in ["bias", "full_review"]:
        # Use LLM to structure the response
        structured_data = llm_client.extract_structured(
            answer_data['answer'],
            schema={
                "population": "Patient population",
                "intervention": "Intervention/treatment",
                "outcomes": "Primary outcomes",
                "sample_size": "Total sample size",
                "key_findings": "Key findings"
            },
            model=model
        )
    
    # Calculate average confidence
    avg_confidence = answer_data['confidence']
    if structured_data:
        confidences = [v.get('confidence', 0.5) for v in structured_data.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    
    return {
        "answer": answer_data['answer'],
        "structured_data": structured_data,
        "average_confidence": avg_confidence,
        "chunks_used": len(relevant_chunks)
    }

@app.post("/export/batch_csv")
async def export_batch_csv(results: List[Dict]):
    """Export batch results to CSV"""
    rows = []
    
    for result in results:
        if result.get('error'):
            rows.append({
                "PDF": result['filename'],
                "Status": "Error",
                "Error": result['error']
            })
        else:
            row = {"PDF": result['filename'], "Status": "Success"}
            
            if result.get('structured_data'):
                for key, value in result['structured_data'].items():
                    row[key] = value.get('text', '')
                    row[f"{key}_confidence"] = f"{value.get('confidence', 0)*100:.1f}%"
            
            row['Average Confidence'] = f"{result.get('average_confidence', 0)*100:.1f}%"
            rows.append(row)
    
    df = pd.DataFrame(rows)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        io.StringIO(stream.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=systematic_review_data.csv"}
    )

@app.post("/export/batch_prisma")
async def export_batch_prisma(results: List[Dict]):
    """Export PRISMA-compliant summary"""
    prisma_text = f"""# Systematic Review Summary (PRISMA Format)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Total Studies Processed: {len(results)}

## Summary of Included Studies

"""
    
    for i, result in enumerate(results, 1):
        if not result.get('error'):
            prisma_text += f"""### Study {i}: {result['filename']}

"""
            if result.get('structured_data'):
                for key, value in result['structured_data'].items():
                    prisma_text += f"**{key.replace('_', ' ').title()}:** {value.get('text', 'Not reported')}\n\n"
            
            prisma_text += f"**Quality Assessment:** Average confidence {result.get('average_confidence', 0)*100:.1f}%\n\n"
            prisma_text += "---\n\n"
    
    # Add summary statistics
    successful = len([r for r in results if not r.get('error')])
    prisma_text += f"""## Summary Statistics

- Total studies screened: {len(results)}
- Studies successfully processed: {successful}
- Studies with errors: {len(results) - successful}

## Data Extraction Summary

This systematic review extracted PICOTT elements, risk of bias assessments, and key outcomes from the included studies.
"""
    
    return {"prisma_text": prisma_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)