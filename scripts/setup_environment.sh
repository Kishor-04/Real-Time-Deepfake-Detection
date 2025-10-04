#!/bin/bash

# Video Deepfake Detection - Environment Setup Script
# Author: Kishor-04
# Date: 2025-01-04

echo "======================================================================"
echo "🎯 Video Deepfake Detection - Environment Setup"
echo "   Author: Kishor-04"
echo "======================================================================"

# Check Python version
echo -e "\n📦 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Create virtual environment
echo -e "\n🔧 Creating virtual environment..."
python3 -m venv venv
echo "   ✓ Virtual environment created"

# Activate virtual environment
echo -e "\n🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"

# Upgrade pip
echo -e "\n⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ Pip upgraded"

# Install dependencies
echo -e "\n📚 Installing dependencies..."
pip install -r requirements.txt --quiet
echo "   ✓ Dependencies installed"

# Verify GPU
echo -e "\n🎮 Checking GPU availability..."
python3 -c "import torch; print(f'   CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'   GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create directory structure
echo -e "\n📁 Creating directory structure..."
mkdir -p data/raw/real
mkdir -p data/raw/fake
mkdir -p data/processed/frames
mkdir -p data/processed/faces
mkdir -p models/pretrained
mkdir -p models/checkpoints
mkdir -p results
mkdir -p runs
echo "   ✓ Directories created"

echo -e "\n======================================================================"
echo "✅ Setup completed successfully!"
echo "======================================================================"
echo -e "\n📋 Next steps:"
echo "   1. Place your videos in data/raw/real/ and data/raw/fake/"
echo "   2. Run: python main.py --mode all"
echo -e "\n======================================================================"