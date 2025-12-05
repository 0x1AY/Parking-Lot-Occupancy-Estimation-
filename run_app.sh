#!/bin/bash
# Quick Start Script for Parking Occupancy Detection Streamlit App

echo "🅿️  Parking Occupancy Detection - Streamlit App"
echo "================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install/upgrade dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-streamlit.txt
echo "✓ Dependencies installed"

# Check if models exist
echo ""
echo "🔍 Checking models..."

MODELS_OK=true

if [ ! -f "datasets/apklot/apklot_stage1/weights/best.pt" ]; then
    echo "❌ Localization model not found: datasets/apklot/apklot_stage1/weights/best.pt"
    MODELS_OK=false
fi

if [ ! -f "parking_runs/yolo11m_parking_augmented2/weights/best.pt" ]; then
    echo "❌ Car detection model not found: parking_runs/yolo11m_parking_augmented2/weights/best.pt"
    MODELS_OK=false
fi

if [ ! -f "parking_runs/yolo11m_multilabel/weights/best.pt" ]; then
    echo "❌ Stall detection model not found: parking_runs/yolo11m_multilabel/weights/best.pt"
    MODELS_OK=false
fi

if [ "$MODELS_OK" = true ]; then
    echo "✓ All models found"
else
    echo ""
    echo "⚠️  Some models are missing. Please ensure all trained models are in place."
    echo "   See README.md for training instructions."
    echo ""
fi

# Launch Streamlit
echo ""
echo "🚀 Launching Streamlit app..."
echo "   URL: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
