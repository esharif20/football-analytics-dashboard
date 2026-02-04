# API Keys Guide

This document explains which API keys are needed for the Football Analysis Dashboard and how to obtain them.

## Summary

| Key | Required? | Used For | Free Tier? |
|-----|-----------|----------|------------|
| **Roboflow API Key** | Optional | Pitch detection fallback | Yes |
| **Gemini API Key** | Optional | AI tactical commentary | Yes |

**Important:** The pipeline works completely without any API keys using the custom-trained models. API keys are only needed for fallback/enhanced features.

---

## 1. Roboflow API Key (Optional)

### Purpose
Used as a **fallback** for pitch keypoint detection when the custom-trained pitch model fails or produces poor results.

### When You Need It
- If pitch detection is failing on your videos
- If you want to compare results with the Roboflow model
- If you're processing videos with unusual camera angles

### How to Get It

1. Go to [roboflow.com](https://roboflow.com)
2. Sign up for a free account
3. Go to Settings → API Keys
4. Copy your API key

### How to Use It

**Option A: Environment Variable**
```bash
export ROBOFLOW_API_KEY=your_key_here
```

**Option B: CLI Flag**
```bash
python main.py --video input.mp4 --mode all --pitch-backend inference
```

The `--pitch-backend inference` flag tells the pipeline to use Roboflow API instead of the local model.

---

## 2. Gemini API Key (Optional)

### Purpose
Used for **AI tactical commentary generation** in the dashboard. The AI analyzes tracking data and generates grounded tactical insights about the match.

### When You Need It
- If you want AI-generated tactical analysis
- If you want natural language commentary about plays

### How to Get It

1. Go to [ai.google.dev](https://ai.google.dev)
2. Click "Get API key"
3. Sign in with your Google account
4. Create a new API key

### How to Use It

Add the key in the dashboard's Settings panel, or set it as an environment variable:

```bash
export GEMINI_API_KEY=your_key_here
```

---

## Running Without API Keys

The pipeline is fully functional without any API keys:

```bash
cd backend
source venv/bin/activate

# Full pipeline with custom models (no API keys needed)
python main.py \
    --source-video-path input_videos/Test6.mp4 \
    --target-video-path output_videos/Test6_output.mp4 \
    --mode all \
    --device auto
```

This will:
- Use the custom-trained player detection model
- Use the custom-trained ball detection model  
- Use the custom-trained pitch detection model
- Generate annotated video, radar video, and analytics JSON

---

## Troubleshooting

### "Roboflow API key not found"
This warning appears if you use `--pitch-backend inference` without setting the API key. Either:
- Set `ROBOFLOW_API_KEY` environment variable
- Use `--pitch-backend ultralytics` to use the local model instead

### "Gemini API error"
If AI commentary fails:
- Check your API key is valid
- Check you haven't exceeded the free tier quota
- The dashboard will still work, just without AI commentary

---

## Security Notes

- Never commit API keys to git
- Use environment variables or `.env` files
- The `.env` file is already in `.gitignore`
- For production, use proper secrets management
