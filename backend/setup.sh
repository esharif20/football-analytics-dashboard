#!/bin/bash
# Football Analysis Pipeline - Setup Script
# Compatible with Mac (Apple Silicon & Intel), Linux, and GPU servers (RunPod/Colab)

set -euo pipefail

echo "=========================================="
echo "Football Analysis Pipeline Setup"
echo "=========================================="

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected: $OS ($ARCH)"

# =============================================================================
# Step 1: Create Python virtual environment
# =============================================================================
echo ""
echo "[1/4] Setting up Python environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# =============================================================================
# Step 2: Install PyTorch (platform-specific)
# =============================================================================
echo ""
echo "[2/4] Installing PyTorch..."

if [ "$OS" = "Darwin" ]; then
    # macOS - use MPS (Metal Performance Shaders) for Apple Silicon
    if [ "$ARCH" = "arm64" ]; then
        echo "Installing PyTorch with MPS support (Apple Silicon)..."
        pip install torch torchvision torchaudio
    else
        echo "Installing PyTorch for Intel Mac..."
        pip install torch torchvision torchaudio
    fi
elif [ "$OS" = "Linux" ]; then
    # Linux - check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "Installing PyTorch (default)..."
    pip install torch torchvision torchaudio
fi

# =============================================================================
# Step 3: Install other dependencies
# =============================================================================
echo ""
echo "[3/4] Installing dependencies..."

pip install ultralytics>=8.0.0
pip install supervision>=0.18.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install transformers>=4.30.0
pip install umap-learn>=0.5.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install Pillow>=10.0.0
pip install python-dotenv>=1.0.0
pip install httpx>=0.24.0
pip install gdown

# =============================================================================
# Step 4: Download models
# =============================================================================
echo ""
echo "[4/4] Downloading models..."

# Create directories
mkdir -p models input_videos output_videos stubs

# Download models from Google Drive
if [ ! -f "models/player_detection.pt" ]; then
    echo "Downloading player detection model..."
    gdown -O "models/player_detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
else
    echo "Player detection model already exists"
fi

if [ ! -f "models/ball_detection.pt" ]; then
    echo "Downloading ball detection model..."
    gdown -O "models/ball_detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
else
    echo "Ball detection model already exists"
fi

if [ ! -f "models/pitch_detection.pt" ]; then
    echo "Downloading pitch detection model..."
    gdown -O "models/pitch_detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
else
    echo "Pitch detection model already exists"
fi

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python main.py --video input_videos/your_video.mp4 --mode all"
echo ""
echo "Device detected:"
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('  Apple Silicon (MPS) - GPU acceleration available')
elif torch.cuda.is_available():
    print(f'  CUDA GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  CPU only')
"
