#!/bin/bash
# Football Analysis Pipeline Setup Script
# Supports Mac, Linux, RunPod, and Google Colab

set -e

echo "=========================================="
echo "Football Analysis Pipeline Setup"
echo "=========================================="

# Detect environment
detect_environment() {
    if [ -n "$RUNPOD_POD_ID" ]; then
        echo "RunPod"
    elif [ -n "$COLAB_GPU" ] || [ -d "/content" ]; then
        echo "Colab"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Mac"
    else
        echo "Linux"
    fi
}

ENV=$(detect_environment)
echo "Detected environment: $ENV"

# Create virtual environment (skip on Colab)
if [ "$ENV" != "Colab" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install PyTorch based on environment
echo "Installing PyTorch..."
if [ "$ENV" == "Mac" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
elif [ "$ENV" == "RunPod" ] || [ "$ENV" == "Colab" ]; then
    # CUDA is available
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    # Linux - check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install pipeline requirements
echo "Installing pipeline dependencies..."
pip install -r requirements.txt

# Download models
echo "Downloading models..."
mkdir -p models

# YOLOv8x for player detection
if [ ! -f "models/yolov8x.pt" ]; then
    echo "Downloading YOLOv8x..."
    wget -q -O models/yolov8x.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt
fi

# Check for custom models
if [ -f "../models/ball_detection.pt" ]; then
    echo "Found custom ball detection model"
    cp ../models/ball_detection.pt models/
fi

if [ -f "../models/pitch_detection.pt" ]; then
    echo "Found custom pitch detection model"
    cp ../models/pitch_detection.pt models/
fi

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Pipeline Configuration
PIPELINE_PORT=8001
DEVICE=auto

# Model Paths
DET_MODEL_PATH=models/yolov8x.pt
BALL_MODEL_PATH=models/ball_detection.pt
PITCH_MODEL_PATH=models/pitch_detection.pt

# Roboflow API (fallback for pitch detection)
ROBOFLOW_API_KEY=

# Dashboard Callback URL
CALLBACK_URL=
EOF
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the pipeline service:"
if [ "$ENV" != "Colab" ]; then
    echo "  source venv/bin/activate"
fi
echo "  python main.py"
echo ""
echo "The service will be available at http://localhost:8001"
echo ""
