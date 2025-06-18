#!/bin/bash

echo "🎨 Starting Streamlit Visual Grounding RAG"
echo "========================================"

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Activate virtual environment
source docling-env/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

echo ""
echo "📍 The app will open in your browser automatically"
echo "📍 If not, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run streamlit_demo.py