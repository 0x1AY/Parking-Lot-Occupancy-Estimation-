#!/bin/bash
# Quick Start Script for Parking Occupancy Detection Streamlit App

echo "🅿️  Parking Occupancy Detection - Streamlit App"
echo "================================================"
echo ""

# Navigate to parent directory
cd "$(dirname "$0")/.."

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
pip install -q -r app/requirements-streamlit.txt
echo "✓ Dependencies installed"

# Check if models exist
echo ""
echo "🔍 Checking for trained models..."
MODELS_OK=true

if [ ! -f "weights/stall_model.pt" ]; then
    echo "   ⚠️  stall_model.pt not found"
    MODELS_OK=false
fi

if [ ! -f "weights/vehicle_model.pt" ]; then
    echo "   ⚠️  vehicle_model.pt not found"
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

streamlit run app/app.py
