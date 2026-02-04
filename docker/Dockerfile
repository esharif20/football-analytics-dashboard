# Football Analysis Dashboard - Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Build the frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy package files
COPY package.json pnpm-lock.yaml ./
COPY client/package.json ./client/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source files
COPY client/ ./client/
COPY shared/ ./shared/
COPY server/ ./server/
COPY drizzle/ ./drizzle/
COPY tsconfig.json vite.config.ts ./

# Build frontend
RUN pnpm build

# =============================================================================
# Stage 2: Python backend with CV pipeline
# =============================================================================
FROM python:3.11-slim AS backend

WORKDIR /app/backend

# Install system dependencies for OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM node:20-slim AS production

WORKDIR /app

# Install pnpm and Python
RUN npm install -g pnpm && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy built frontend and server
COPY --from=frontend-builder /app/dist ./dist
COPY --from=frontend-builder /app/node_modules ./node_modules
COPY --from=frontend-builder /app/package.json ./
COPY --from=frontend-builder /app/server ./server
COPY --from=frontend-builder /app/shared ./shared
COPY --from=frontend-builder /app/drizzle ./drizzle

# Copy Python backend
COPY --from=backend /app/backend ./backend
COPY --from=backend /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Create directories for data persistence
RUN mkdir -p /app/data /app/uploads /app/output

# Environment variables
ENV NODE_ENV=production
ENV DATABASE_URL=file:/app/data/football.db
ENV UPLOAD_DIR=/app/uploads
ENV OUTPUT_DIR=/app/output

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start command
CMD ["node", "dist/server/index.js"]
