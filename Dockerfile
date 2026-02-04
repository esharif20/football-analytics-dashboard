# Football Analytics Dashboard
# Single Dockerfile - builds frontend and runs FastAPI backend

FROM python:3.11-slim

# Install Node.js for building frontend
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy frontend and build it
COPY frontend/package.json frontend/pnpm-lock.yaml* ./frontend/
RUN cd frontend && pnpm install --frozen-lockfile 2>/dev/null || pnpm install

COPY frontend ./frontend
RUN cd frontend && pnpm build

# Install Python dependencies
COPY backend/api/requirements.txt ./backend/api/
RUN pip install --no-cache-dir -r backend/api/requirements.txt

# Copy backend code
COPY backend ./backend

# Create data directories
RUN mkdir -p backend/data/uploads backend/data/outputs

# Expose port
EXPOSE 8000

# Set working directory to backend
WORKDIR /app/backend

# Run FastAPI
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
