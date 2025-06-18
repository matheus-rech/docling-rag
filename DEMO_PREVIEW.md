# 🔬 Visual Grounding RAG - Demo Preview

## 🌐 FastAPI Interface (http://localhost:8000)

### Main Features:
- **Single PDF Processing** with visual grounding
- **Model Selection**: DeepSeek (medical), Mistral, Llama3
- **Query Types**: PICOTT, Risk of Bias, Outcomes, Custom
- **Visual Output**: Red bounding boxes on PDF pages
- **Export**: CSV and PRISMA formats

### Interface Preview:
```
┌─────────────────────────────────────────────────────┐
│ 🔬 Medical RAG - Systematic Review Assistant        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📄 Upload Medical Paper (PDF)                       │
│ [Choose File] sample.pdf                            │
│                                                     │
│ 🤖 Select Model:                                    │
│ [▼ DeepSeek (Recommended for Medical)]             │
│                                                     │
│ 🔍 Query Type:                                      │
│ [▼ Extract PICOTT]                                  │
│                                                     │
│ [🚀 Process Document]                               │
│                                                     │
├─────────────────────────────────────────────────────┤
│ 📊 Results                                          │
│                                                     │
│ Structured Extraction:                              │
│ Population: 65 patients with cerebellar infarction  │
│            (confidence: 92.3%)                      │
│ Intervention: SDC surgery                           │
│            (confidence: 88.5%)                      │
│ Comparator: Conservative treatment                  │
│            (confidence: 85.1%)                      │
│                                                     │
│ 📍 Source Location:                                 │
│ Page 1, Position: x=99, y=627, width=392, h=35     │
│                                                     │
│ [Download CSV] [Show PRISMA]                        │
└─────────────────────────────────────────────────────┘
```

## 🎨 Streamlit Interface (streamlit run streamlit_visual_app.py)

### Visual Grounding Display:
```
┌─────────────────────────────────────┐┌─────────────────────────────────────┐
│ 📄 Upload & Query                   ││ 📊 Results                          │
├─────────────────────────────────────┤├─────────────────────────────────────┤
│ Upload medical paper (PDF)          ││ Answer:                             │
│ [📁 sample.pdf]                     ││ The study included 65 patients      │
│                                     ││ with cerebellar infarction who      │
│ Selected prompt:                    ││ underwent preventive SDC surgery... │
│ Extract the PICOTT (Population,     ││                                     │
│ Intervention, Comparator, Outcome,  ││ Confidence Score: 87.5% ████████░   │
│ Time, Type of Study) from this...   ││                                     │
│                                     ││ 📍 Visual Grounding                 │
│ [🔍 Extract with Visual Grounding]  ││ Found on page 1                     │
│                                     ││                                     │
│                                     ││ ┌─────────────────────────────┐     │
│                                     ││ │  [PDF PAGE WITH RED BOX]    │     │
│                                     ││ │                             │     │
│                                     ││ │  ┌─────────────────┐        │     │
│                                     ││ │  │ Highlighted text │        │     │
│                                     ││ │  └─────────────────┘        │     │
│                                     ││ │                             │     │
│                                     ││ └─────────────────────────────┘     │
│                                     ││                                     │
│                                     ││ [📊 Download CSV] [📄 Download PRISMA]│
└─────────────────────────────────────┘└─────────────────────────────────────┘
```

## 📚 Batch Processing (http://localhost:8001)

### Features:
- Upload multiple PDFs at once
- Progress bar for batch processing
- Comparison grid view
- Export all results to CSV/PRISMA

### Batch Results View:
```
┌─────────────────────────────────────────────────────────────────────┐
│ 📋 Extraction Results                                               │
│ ✅ Successfully processed: 5/5 PDFs                                 │
├────────────────┬─────────────────────────────┬──────────┬──────────┤
│ PDF            │ Extracted Data              │ Confidence│ Actions  │
├────────────────┼─────────────────────────────┼──────────┼──────────┤
│ study1.pdf     │ Population: 120 patients... │ 91.2%    │ [Details]│
│ study2.pdf     │ Population: 85 patients...  │ 88.7%    │ [Details]│
│ study3.pdf     │ Population: 200 patients... │ 93.5%    │ [Details]│
│ trial_2023.pdf │ Population: 150 patients... │ 86.9%    │ [Details]│
│ review.pdf     │ Error: Not a clinical trial │ -        │ [Details]│
├────────────────┴─────────────────────────────┴──────────┴──────────┤
│ [📥 Export to CSV] [📄 Export PRISMA Summary] [📊 Compare Studies]  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start Test

1. **Activate environment & run test:**
```bash
source docling-env/bin/activate
python test_system.py
```

2. **Expected test output:**
```
🔬 Visual Grounding RAG - System Test
==================================================
🔍 Testing imports...
  ✅ Docling
  ✅ FAISS
  ✅ Sentence Transformers
  ✅ LangChain
  ✅ FastAPI
  ✅ Streamlit
  ✅ Pandas
  ✅ Matplotlib

🤖 Testing Ollama...
  ✅ Ollama is running
  ✅ DeepSeek model found

📄 Testing PDF processing...
  ✅ Sample PDF found
  ⏳ Processing PDF (this may take a moment)...
  ✅ PDF processed: 71 chunks created

🔍 Testing vector store...
  ✅ Vector search working: 2 results found

✅ All tests passed!

🎯 Running quick demo...
==================================================
📄 Processing sample.pdf...
✅ Created 71 chunks

🔍 Query: 'What is the patient diagnosis and treatment?'

📊 Found 3 relevant chunks:

Chunk 1:
Text: P atients with cerebellar infarction should not be neglected because they can experience sudden clinical deterioration from  cerebellar  swe...
📍 Location: Page 1

Chunk 2:
Text: Myeong Jin Kim, MD; Sang Kyu Park, MD; Jihye Song, MD; Se-yang Oh, MD; Yong Cheol Lim, MD; Sook Yong Sim, MD, PhD; Yong Sam Shin, MD, PhD;...
📍 Location: Page 1

✅ Demo complete!

🚀 Ready to start the web interface!

Choose an option:
1. FastAPI interface (recommended)
2. Streamlit interface
3. Batch processing interface
4. Exit

Enter your choice (1-4): _
```

## 📸 Key Visual Features

1. **Visual Grounding**: Shows exact location in PDF with red bounding boxes
2. **Confidence Scores**: Color-coded (green >70%, orange 50-70%, red <50%)
3. **PICOTT Extraction**: Structured medical data with field-level confidence
4. **Batch Comparison**: Side-by-side view of multiple studies
5. **Export Options**: CSV for data analysis, PRISMA for systematic reviews

The system is now ready for processing medical PDFs with visual grounding!