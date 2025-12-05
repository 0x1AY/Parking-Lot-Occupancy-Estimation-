#!/bin/bash
# Install Dependencies for Streamlit App

echo "📦 Installing Parking Occupancy Detection Dependencies"
echo "======================================================"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core requirements
echo ""
echo "📥 Installing core dependencies (PyTorch, OpenCV, etc.)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "📥 Installing other core dependencies..."
pip install opencv-python Pillow numpy pandas requests tqdm pyyaml ultralytics

# Install Streamlit
echo ""
echo "📥 Installing Streamlit..."
pip install streamlit

echo ""
echo "✅ Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  python test_streamlit_pipeline.py"
echo ""
echo "To launch the app, run:"
echo "  streamlit run app.py"
