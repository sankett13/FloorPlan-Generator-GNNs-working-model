#!/bin/bash
# Setup script for Floor Plan Generator

echo "🏠 Floor Plan Generator - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Navigate to project directory
cd "/Users/sanketpatel/Desktop/Floor Plan Try/Floor_Plan_Generation_using_GNNs"

# Create Outputs directory if it doesn't exist
echo ""
echo "Creating Outputs directory..."
mkdir -p Outputs

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install PyTorch Geometric dependencies
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Check if model exists
echo ""
echo "Checking for model checkpoint..."
if [ -f "GAT-Net_model/checkpoints/GAT-Net_v3_UnScalled.pt" ]; then
    echo "✅ Model checkpoint found!"
else
    echo "❌ Model checkpoint NOT found at: GAT-Net_model/checkpoints/GAT-Net_v3_UnScalled.pt"
    echo "   Please ensure the model file exists before running the application."
fi

echo ""
echo "========================================"
echo "Setup complete! 🎉"
echo ""
echo "To run the application:"
echo "  streamlit run app.py"
echo ""
