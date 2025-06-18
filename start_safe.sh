#!/bin/bash

echo "🔬 Starting Visual Grounding RAG (Safe Mode)"
echo "=========================================="

# Set environment variables to avoid multiprocessing issues on Mac
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Activate virtual environment
source docling-env/bin/activate

echo ""
echo "Choose an interface to start:"
echo "1. FastAPI (Single PDF with visual grounding)"
echo "2. Streamlit (User-friendly interface)"
echo "3. Batch Processing (Multiple PDFs)"
echo "4. Run basic tests"
echo "5. Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Starting FastAPI interface..."
        echo "📍 Open http://localhost:8000 in your browser"
        echo ""
        python -m uvicorn app:app --reload --workers 1
        ;;
    2)
        echo "🚀 Starting Streamlit interface..."
        echo ""
        streamlit run streamlit_visual_app.py
        ;;
    3)
        echo "🚀 Starting Batch Processing interface..."
        echo "📍 Open http://localhost:8001 in your browser"
        echo ""
        python -m uvicorn app_batch:app --reload --workers 1 --port 8001
        ;;
    4)
        echo "🧪 Running basic tests..."
        echo ""
        python quick_test.py
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac