# Football Analysis Pipeline - Docker Image
# Optimized for RunPod and GPU environments

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8x model
RUN mkdir -p models && \
    wget -q -O models/yolov8x.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt

# Copy application code
COPY . .

# Make setup script executable
RUN chmod +x setup.sh

# Expose port
EXPOSE 8001

# Environment variables
ENV PIPELINE_PORT=8001
ENV DEVICE=cuda

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start the service
CMD ["python", "main.py"]
