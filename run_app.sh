#!/bin/bash

# Set environment variables to avoid issues
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Activate virtual environment
source docling-env/bin/activate

echo "🔬 Starting Visual Grounding RAG Server"
echo "======================================"
echo ""
echo "📍 The server will start at: http://localhost:8000"
echo ""
echo "Please wait a moment for the server to start..."
echo "Then open http://localhost:8000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1