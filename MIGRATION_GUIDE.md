# Football Analytics Dashboard - Independent Deployment Guide

This document provides comprehensive instructions for running the Football Analytics Dashboard independently from Manus infrastructure. It is designed to be readable by both humans and CLI-based LLMs (Codex, Claude Code, etc.) for automated setup assistance.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Manus-Specific Dependencies](#manus-specific-dependencies)
3. [Local Development Setup (macOS)](#local-development-setup-macos)
4. [Production Deployment](#production-deployment)
5. [RunPod Worker Setup](#runpod-worker-setup)
6. [Service Replacements](#service-replacements)
7. [Environment Variables Reference](#environment-variables-reference)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The Football Analytics Dashboard consists of three main components:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 19 + Vite + Tailwind CSS 4 | User interface for uploading videos and viewing analysis results |
| **Backend** | Express + tRPC + Drizzle ORM | API server handling authentication, database operations, and file storage |
| **Worker** | Python + PyTorch + Ultralytics | GPU-accelerated computer vision pipeline running on RunPod |

The system flow operates as follows: users upload match videos through the frontend, which stores them via the backend to cloud storage. The Python worker polls for pending analyses, downloads videos, runs the CV pipeline (player detection, ball tracking, pitch homography), and uploads annotated videos back to storage. Results are saved to the database and displayed in the dashboard.

---

## Manus-Specific Dependencies

The following table lists all Manus-specific services and their required replacements:

| Service | Environment Variables | Purpose | Replacement Options |
|---------|----------------------|---------|---------------------|
| **Manus OAuth** | `VITE_APP_ID`, `OAUTH_SERVER_URL`, `VITE_OAUTH_PORTAL_URL` | User authentication | Auth0, Clerk, NextAuth.js, or custom JWT |
| **Manus Forge Storage** | `BUILT_IN_FORGE_API_URL`, `BUILT_IN_FORGE_API_KEY` | S3-compatible file storage | AWS S3, Cloudflare R2, MinIO, or local filesystem |
| **Manus Forge LLM** | `BUILT_IN_FORGE_API_URL`, `BUILT_IN_FORGE_API_KEY` | AI commentary generation | OpenAI API, Anthropic Claude, or local Ollama |
| **TiDB Cloud** | `DATABASE_URL` | MySQL-compatible database | MySQL, PostgreSQL, PlanetScale, or SQLite |
| **Manus Analytics** | `VITE_ANALYTICS_ENDPOINT`, `VITE_ANALYTICS_WEBSITE_ID` | Usage tracking | Plausible, Umami, or remove entirely |
| **Model CDN** | Hardcoded URLs in `worker.py` | Custom YOLO model hosting | Your own S3 bucket or local files |

---

## Local Development Setup (macOS)

### Prerequisites

Before beginning, ensure you have the following installed on your Mac:

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Node.js 22+ and pnpm
brew install node
npm install -g pnpm

# Install Python 3.11+ (for local worker testing)
brew install python@3.11

# Install MySQL (optional - can use SQLite instead)
brew install mysql
brew services start mysql
```

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# Install Node.js dependencies
pnpm install

# Install backend dependencies
cd backend && pnpm install && cd ..
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root with the following configuration for local development:

```bash
# .env - Local Development Configuration

# Enable local development mode (bypasses Manus OAuth)
LOCAL_DEV_MODE=true

# Database - Use local MySQL or SQLite
# For MySQL:
DATABASE_URL="mysql://root:password@localhost:3306/football_dashboard"
# For SQLite (simpler):
# DATABASE_URL="file:./local.db"

# JWT Secret (generate with: openssl rand -base64 32)
JWT_SECRET="your-random-secret-key-here"

# Local storage directory for uploaded files
LOCAL_STORAGE_DIR="./uploads"

# Optional: OpenAI API key for AI commentary (can skip if not needed)
OPENAI_API_KEY="sk-your-openai-key"

# Disable Manus-specific services
BUILT_IN_FORGE_API_URL=""
BUILT_IN_FORGE_API_KEY=""
OAUTH_SERVER_URL=""
VITE_APP_ID=""
```

### Step 3: Initialize Database

```bash
cd backend

# Generate and run migrations
pnpm db:push

# Verify database tables were created
pnpm db:studio  # Opens Drizzle Studio to inspect database
```

### Step 4: Start Development Server

```bash
# From project root
pnpm dev
```

The application will be available at `http://localhost:3000`. When `LOCAL_DEV_MODE=true`, you are automatically logged in as a local admin user without needing OAuth.

### Step 5: Local File Storage

When running locally, uploaded files are stored in the `./uploads` directory instead of cloud storage. The backend automatically serves these files at `/uploads/*`. No additional configuration is required.

---

## Production Deployment

For production deployment without Manus infrastructure, you need to set up your own services.

### Option A: AWS-Based Deployment

This configuration uses AWS services as replacements:

```bash
# .env.production

# Database - Use AWS RDS MySQL or PlanetScale
DATABASE_URL="mysql://user:pass@your-rds-endpoint.amazonaws.com:3306/football_dashboard"

# S3 Storage
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
S3_BUCKET_NAME="your-football-analytics-bucket"

# Authentication - Use Auth0
AUTH0_DOMAIN="your-tenant.auth0.com"
AUTH0_CLIENT_ID="your-client-id"
AUTH0_CLIENT_SECRET="your-client-secret"

# LLM - Use OpenAI directly
OPENAI_API_KEY="sk-your-openai-key"

# JWT Secret
JWT_SECRET="your-production-secret"
```

### Option B: Self-Hosted Deployment

For complete self-hosting with minimal external dependencies:

```bash
# .env.selfhosted

# Local MySQL database
DATABASE_URL="mysql://root:password@localhost:3306/football_dashboard"

# MinIO for S3-compatible storage
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_BUCKET="football-analytics"

# Local Ollama for LLM
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3"

# Simple JWT auth (no OAuth)
LOCAL_DEV_MODE=true
JWT_SECRET="your-secret"
```

---

## RunPod Worker Setup

The CV pipeline worker runs on RunPod with GPU acceleration. This section remains largely unchanged from the current setup, as the worker communicates with your dashboard via HTTP APIs.

### Worker Configuration

The worker needs to know where your dashboard is hosted. Update the `DASHBOARD_URL` environment variable:

```bash
# On RunPod
export DASHBOARD_URL="https://your-dashboard-domain.com"
export ROBOFLOW_API_KEY="your-roboflow-key"  # For pitch detection
```

### Model Hosting

The custom YOLO models are currently hosted on Manus CDN. You have two options:

**Option 1: Host models on your own S3 bucket**

Upload the model files to your S3 bucket and update `worker.py`:

```python
# backend/pipeline/worker.py - Update MODEL_URLS

MODEL_URLS = {
    "player_detection.pt": "https://your-bucket.s3.amazonaws.com/models/player_detection.pt",
    "ball_detection.pt": "https://your-bucket.s3.amazonaws.com/models/ball_detection.pt",
    "pitch_detection.pt": "https://your-bucket.s3.amazonaws.com/models/pitch_detection.pt",
}
```

**Option 2: Pre-download models to RunPod volume**

Download models once and store them on a persistent RunPod volume:

```bash
# On RunPod, download models manually
cd /football-analytics-dashboard/backend/pipeline/models
wget -O player_detection.pt "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/XAzhckYwibJeQRhg.pt"
wget -O ball_detection.pt "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/NiUwnYcULyjvIBhr.pt"
wget -O pitch_detection.pt "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/pSlXgeDoBtmXQHTJ.pt"
```

### Running the Worker

```bash
cd /football-analytics-dashboard/backend/pipeline
export DASHBOARD_URL="https://your-dashboard.com"
export ROBOFLOW_API_KEY="your-key"
python worker.py
```

The worker will poll your dashboard for pending analyses and process them automatically.

---

## Service Replacements

### Replacing Manus OAuth with Auth0

To replace Manus OAuth with Auth0, modify the following files:

**1. Create `backend/server/_core/auth0.ts`:**

```typescript
import { Auth0Client } from '@auth0/auth0-spa-js';

export const auth0Config = {
  domain: process.env.AUTH0_DOMAIN!,
  clientId: process.env.AUTH0_CLIENT_ID!,
  clientSecret: process.env.AUTH0_CLIENT_SECRET!,
  redirectUri: `${process.env.APP_URL}/api/oauth/callback`,
};

// Implement token exchange and user info retrieval
// following Auth0's Node.js SDK documentation
```

**2. Update `backend/server/_core/oauth.ts`:**

Replace the Manus SDK calls with Auth0 SDK calls for token exchange and user info retrieval.

**3. Update frontend login URL:**

Modify `client/src/const.ts` to generate Auth0 login URLs instead of Manus OAuth URLs.

### Replacing Manus Storage with AWS S3

The storage module already supports local file storage. To use AWS S3 instead:

**1. Install AWS SDK:**

```bash
cd backend && pnpm add @aws-sdk/client-s3 @aws-sdk/s3-request-presigner
```

**2. Update `backend/server/storage.ts`:**

```typescript
import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

export async function storagePut(
  relKey: string,
  data: Buffer | Uint8Array | string,
  contentType = "application/octet-stream"
): Promise<{ key: string; url: string }> {
  const command = new PutObjectCommand({
    Bucket: process.env.S3_BUCKET_NAME,
    Key: relKey,
    Body: data,
    ContentType: contentType,
  });
  
  await s3Client.send(command);
  
  const url = `https://${process.env.S3_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${relKey}`;
  return { key: relKey, url };
}
```

### Replacing Manus LLM with OpenAI

**1. Install OpenAI SDK:**

```bash
cd backend && pnpm add openai
```

**2. Update `backend/server/_core/llm.ts`:**

```typescript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function invokeLLM(params: InvokeParams): Promise<InvokeResult> {
  const response = await openai.chat.completions.create({
    model: 'gpt-4-turbo-preview',
    messages: params.messages,
    tools: params.tools,
    tool_choice: params.toolChoice,
    max_tokens: params.maxTokens || 4096,
  });
  
  return response as InvokeResult;
}
```

---

## Environment Variables Reference

Complete list of all environment variables and their purposes:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | - | MySQL/PostgreSQL connection string |
| `JWT_SECRET` | Yes | - | Secret key for signing JWT tokens |
| `LOCAL_DEV_MODE` | No | `false` | Enable local development mode (bypasses OAuth) |
| `LOCAL_STORAGE_DIR` | No | `./uploads` | Directory for local file storage |
| `OPENAI_API_KEY` | No | - | OpenAI API key for AI commentary |
| `AWS_REGION` | No | - | AWS region for S3 |
| `AWS_ACCESS_KEY_ID` | No | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | No | - | AWS secret key |
| `S3_BUCKET_NAME` | No | - | S3 bucket name |
| `AUTH0_DOMAIN` | No | - | Auth0 tenant domain |
| `AUTH0_CLIENT_ID` | No | - | Auth0 application client ID |
| `AUTH0_CLIENT_SECRET` | No | - | Auth0 application client secret |
| `ROBOFLOW_API_KEY` | No | - | Roboflow API key for pitch detection |
| `DASHBOARD_URL` | Worker | `http://localhost:3000` | Dashboard URL for worker API calls |

---

## Troubleshooting

### Common Issues

**Issue: "Storage proxy credentials missing" error**

This occurs when `BUILT_IN_FORGE_API_URL` and `BUILT_IN_FORGE_API_KEY` are not set and `LOCAL_DEV_MODE` is not enabled. Solution: Either set `LOCAL_DEV_MODE=true` for local development, or configure your own S3 storage.

**Issue: Database connection fails**

Ensure your `DATABASE_URL` is correctly formatted. For MySQL: `mysql://user:password@host:port/database`. For local development, ensure MySQL is running: `brew services start mysql`.

**Issue: Worker cannot connect to dashboard**

Verify the `DASHBOARD_URL` environment variable points to your dashboard's public URL. Ensure the dashboard is accessible from the RunPod network.

**Issue: Models fail to download**

If the Manus CDN URLs become unavailable, download the models manually and place them in `backend/pipeline/models/`. The worker will skip downloading if files already exist.

**Issue: OAuth redirect fails**

When replacing Manus OAuth, ensure your callback URL is correctly configured in your OAuth provider (Auth0, etc.) and matches the URL in your code.

### Getting Help

For issues specific to the CV pipeline, check the `backend/pipeline/src/` directory for detailed module documentation. For frontend issues, the React components are in `client/src/pages/` and `client/src/components/`.

---

## Quick Start Commands Summary

```bash
# Local development (macOS)
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard
pnpm install
echo "LOCAL_DEV_MODE=true" > .env
echo "DATABASE_URL=mysql://root@localhost:3306/football" >> .env
echo "JWT_SECRET=$(openssl rand -base64 32)" >> .env
cd backend && pnpm db:push && cd ..
pnpm dev

# RunPod worker
cd /football-analytics-dashboard/backend/pipeline
export DASHBOARD_URL="https://your-dashboard.com"
python worker.py
```

---

*Document Version: 1.0*  
*Last Updated: February 2026*  
*Author: Manus AI*
