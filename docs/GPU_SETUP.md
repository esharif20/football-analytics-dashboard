# GPU Access Guide for Football Analysis Pipeline

This guide covers pay-as-you-go GPU options suitable for a BSc dissertation project. All options require no upfront commitment and charge only for actual usage.

## Quick Comparison

| Platform | Cost | GPU Options | Best For |
|----------|------|-------------|----------|
| **Google Colab Pro** | £9.99/month | T4, V100, A100 | Easiest setup, familiar interface |
| **RunPod** | ~$0.20-0.50/hr | RTX 3090, A100 | Best value, Docker support |
| **Vast.ai** | ~$0.15-0.40/hr | Various | Cheapest, community GPUs |
| **Lambda Labs** | ~$0.50-1.50/hr | A100, H100 | Enterprise-grade |

## Recommended: Google Colab Pro (Easiest)

You already have Colab Pro - this is the simplest option for your dissertation.

### Setup Steps

1. Open `backend/Football_Analysis_Pipeline.ipynb` in Colab
2. Go to **Runtime → Change runtime type → GPU**
3. Select **T4** (free tier) or **V100/A100** (Pro)
4. Run all cells

### Estimated Processing Times (30-second video)

| GPU | Detection | Tracking | Team | Pitch | Total |
|-----|-----------|----------|------|-------|-------|
| T4 | ~60s | ~5s | ~30s | ~20s | ~2 min |
| V100 | ~30s | ~3s | ~20s | ~15s | ~1.5 min |
| A100 | ~20s | ~2s | ~15s | ~10s | ~1 min |

---

## Alternative: RunPod (Best Value)

RunPod offers on-demand GPU instances with Docker support - great for reproducible runs.

### Setup Steps

1. Create account at [runpod.io](https://runpod.io)
2. Add credits (minimum $10)
3. Deploy a pod:
   - Template: **PyTorch 2.0**
   - GPU: **RTX 3090** (~$0.20/hr) or **A100** (~$0.80/hr)
   - Disk: **20GB**

4. SSH into the pod and run:

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/football-dashboard.git
cd football-dashboard/backend

# Install dependencies
pip install -r requirements.txt

# Download models (or upload your own)
./setup.sh

# Run pipeline
python main.py \
    --source-video-path input_videos/your_video.mp4 \
    --target-video-path output_videos/result.mp4 \
    --mode all \
    --device cuda
```

### Cost Estimate

For a 30-second video on RTX 3090:
- Processing time: ~2.5 minutes
- Cost: ~$0.01 per video

For a full match (90 minutes):
- Processing time: ~7-8 hours
- Cost: ~$1.50-2.00

---

## Alternative: Vast.ai (Cheapest)

Vast.ai is a marketplace for renting community GPUs - often 30-50% cheaper than RunPod.

### Setup Steps

1. Create account at [vast.ai](https://vast.ai)
2. Add credits (minimum $5)
3. Search for instances:
   - Filter: **RTX 3090**, **PyTorch**, **>20GB disk**
   - Sort by: **$/hr**

4. Launch instance and SSH in:

```bash
# Same setup as RunPod
git clone https://github.com/YOUR_USERNAME/football-dashboard.git
cd football-dashboard/backend
pip install -r requirements.txt
python main.py --source-video-path video.mp4 --target-video-path output.mp4 --mode all
```

### Pros/Cons

✅ Cheapest option (community GPUs)
✅ Wide variety of hardware
⚠️ Less reliable (community hosts)
⚠️ Variable availability

---

## Using Docker (RunPod/Vast.ai)

For reproducible deployments, use the included Dockerfile:

```bash
# Build image
cd backend
docker build -t football-pipeline .

# Run with GPU
docker run --gpus all -v $(pwd)/input_videos:/app/input_videos \
    -v $(pwd)/output_videos:/app/output_videos \
    football-pipeline \
    python main.py --source-video-path /app/input_videos/video.mp4 \
    --target-video-path /app/output_videos/output.mp4 --mode all
```

---

## Local Mac (No GPU Rental)

If you want to test locally on your Mac without GPU rental:

```bash
cd backend
./setup.sh
source venv/bin/activate

# Run with MPS (Apple Silicon) or CPU
python main.py \
    --source-video-path input_videos/Test6.mp4 \
    --target-video-path output_videos/Test6_output.mp4 \
    --mode all \
    --device auto  # auto-detects MPS on Apple Silicon
```

**Expected times on M1/M2 Mac:**
- 30-second video: ~5-7 minutes
- Full match: ~15-20 hours (not recommended)

---

## Cost Summary for Dissertation

Assuming you process 10-20 short clips for testing and 2-3 full matches:

| Platform | Estimated Total Cost |
|----------|---------------------|
| Colab Pro | £9.99/month (unlimited) |
| RunPod | ~$5-10 total |
| Vast.ai | ~$3-7 total |

**Recommendation:** Start with Colab Pro since you already have it. Use RunPod for longer videos or batch processing.

---

## Tips for Dissertation

1. **Cache your stubs** - The pipeline caches intermediate results. Re-running on the same video skips detection/tracking.

2. **Use `--mode radar`** - If you only need the 2D pitch view, skip the annotated video rendering (2x faster).

3. **Process clips, not full matches** - For analysis, 30-60 second clips are usually sufficient.

4. **Download outputs immediately** - Cloud instances may be terminated. Download results after each run.

5. **Keep a cost log** - Track your GPU spending for the dissertation methodology section.
