# Claude Memory - Visual Grounding RAG Project

## Project Overview
Building a Visual Grounding RAG system for medical/scientific papers that shows exactly WHERE in PDFs the answers come from.

## Key Components Built

### 1. Core Architecture
- **PDF Processing**: Using Docling for layout-aware parsing
- **Vector Search**: FAISS for semantic similarity
- **LLM Integration**: Ollama with DeepSeek/Mistral/Llama3
- **Visual Grounding**: Bounding boxes on PDF pages

### 2. Medical Features
- **PICOTT Extraction**: Population, Intervention, Comparator, Outcome, Time, Type
- **Risk of Bias**: Cochrane framework assessment
- **Confidence Scoring**: Per-field validation (0-100%)
- **Systematic Review**: PRISMA-compliant exports

### 3. Interfaces Created
- **app.py**: FastAPI single PDF processing with visual grounding
- **app_batch.py**: Batch processing for multiple PDFs
- **streamlit_visual_app.py**: User-friendly Streamlit interface

## Technical Details

### Visual Grounding Implementation
```python
# Bounding box extraction from Docling chunks
bbox = chunk.meta.doc_items[0].prov[0].bbox
page_no = chunk.meta.doc_items[0].prov[0].page_no

# Coordinate conversion (bottom-left to top-left origin)
x = bbox.l
y = img_height - bbox.t
width = bbox.r - bbox.l
height = bbox.t - bbox.b
```

### Confidence Scoring
- Implemented at answer level AND field level
- Color coding: Green >70%, Orange 50-70%, Red <50%
- Helps validate medical data extraction quality

## Current Status
- Core functionality:  Complete
- FastAPI installed:  
- Streamlit: ó Installing
- DeepSeek model: ó Downloading (9% complete)
- Sample PDF:  Medical paper on cerebellar infarction

## Quick Test Commands
```bash
# Basic test (works now)
python test_docling.py

# Full system test (after installs)
python test_system.py

# Start interfaces
python app.py                    # FastAPI
streamlit run streamlit_visual_app.py  # Streamlit
```

## Key Files
- `core/`: Modular backend components
- `medical/`: Domain-specific extractors
- `test_system.py`: Complete testing suite
- `PROJECT_STATUS.md`: Detailed progress tracking

## User's Context
- Mac with Apple Silicon
- Interested in neurosurgery research
- Needs systematic review tools
- Prefers local processing (privacy)