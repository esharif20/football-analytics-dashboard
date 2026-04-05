# Agent Guidelines for Football Analytics Dashboard

This document provides guidelines for agents working on this codebase.

---

## Project Overview

Full-stack sports analytics platform — upload football match footage, get real-time player tracking, team classification, heatmaps, pass networks, and AI tactical commentary.

- **Frontend**: React 19 + Vite 6 + TypeScript + Tailwind CSS 4
- **Backend**: FastAPI + SQLAlchemy async + PostgreSQL (Supabase)
- **Pipeline**: Python CV pipeline (YOLOv8, ByteTrack, etc.)

---

## Build / Lint / Test Commands

### Frontend (React + TypeScript)

```bash
# Install dependencies
cd frontend && pnpm install

# Development
pnpm dev          # Start Vite dev server (port 5173)
pnpm build        # Production build
pnpm preview      # Preview production build
pnpm check        # TypeScript type checking (tsc --noEmit)
pnpm lint         # ESLint
pnpm test         # Vitest unit tests
```

### Backend (FastAPI)

```bash
# Install dependencies
cd backend/api && pip install -r requirements.txt

# Development
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# With custom reload directory
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir api

# Lint
cd backend && ruff check api/

# Tests
cd backend && python -m pytest api/tests/          # API unit tests
cd backend && python -m pytest tests/              # Pipeline tests
```

### Root Commands

```bash
# Convenience scripts
pnpm dev          # Frontend only
pnpm dev:api      # API only (backend)
pnpm build        # Build frontend
pnpm test:e2e     # Playwright E2E tests
```

---

## Code Style Guidelines

### General Principles

- Keep code concise and readable
- Avoid unnecessary comments (unless explaining complex logic)
- Use modern JavaScript/TypeScript and Python features

---

### Frontend (React + TypeScript)

#### Imports

- Use path aliases (`@/*`) for internal imports:
  ```tsx
  import { Button } from "@/components/ui/button";
  import { useAuth } from "@/hooks/useAuth";
  import { AnalysisResponse } from "@/types";
  ```
- Group imports: external libs → internal aliases → relative imports
- Use named exports for components and utilities

#### TypeScript

- **Strict mode enabled** — all compiler strict options are on
- Always define prop types for components:
  ```tsx
  interface Props {
    children: ReactNode;
    className?: string;
  }
  ```
- Use `type` for simple types, `interface` for object shapes
- Avoid `any` — use `unknown` when type is truly unknown

#### Naming Conventions

- **Components**: PascalCase (`Dashboard.tsx`, `PlayerCard.tsx`)
- **Hooks**: camelCase with `use` prefix (`useWebSocket.ts`, `useAuth.ts`)
- **Utilities**: camelCase (`formatDate.ts`, `cn.ts`)
- **Constants**: UPPER_SNAKE_CASE for values (`const MAX_UPLOAD_SIZE = 100 * 1024 * 1024`)
- **Files**: kebab-case for non-component files (`api-client.ts`, `utils.ts`)

#### React Patterns

- Use functional components with hooks
- Prefer composition over inheritance
- Extract reusable logic into custom hooks
- Use error boundaries for graceful error handling

#### Styling

- Use Tailwind CSS utility classes
- Use `cn()` utility (clsx + tailwind-merge) for conditional classes:
  ```tsx
  import { cn } from "@/lib/utils";

  <div className={cn("base-class", condition && "conditional-class")} />
  ```
- Use shadcn/ui components from `@/components/ui/*`

#### State & Data Fetching

- Use React Query (`@tanstack/react-query`) for server state
- Use React Context for global client state (theme, auth)
- Keep components small and focused

#### Error Handling

- Use ErrorBoundary components for catching React errors
- Handle API errors gracefully with user feedback
- Use TypeScript error types for typed errors

---

### Backend (Python + FastAPI)

#### Python Version

- Python 3.11+

#### Imports

- Standard library → third-party → local
  ```python
  import os
  from pathlib import Path
  from contextlib import asynccontextmanager

  from fastapi import FastAPI
  from sqlalchemy.ext.asyncio import AsyncSession

  from .config import settings
  from .models import User
  ```

#### Type Hints

- Use Python 3.10+ type hints (no `from __future__ import annotations`)
- Use `Optional[T]` instead of `T | None`
- Annotate async functions properly:
  ```python
  async def get_user(user_id: int) -> User | None:
      ...
  ```

#### Naming Conventions

- **Functions/methods**: snake_case (`get_user`, `create_video`)
- **Classes**: PascalCase (`UserModel`, `VideoResponse`)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: `_private_method`

#### FastAPI Patterns

- Use dependency injection for DB sessions, auth
- Define Pydantic schemas for request/response validation
- Use async/await throughout
- Group routes in routers (`api/routers/`)

#### Error Handling

- Use FastAPI's `HTTPException` for HTTP errors
- Return proper HTTP status codes
- Use Pydantic validation errors for input validation

#### Database

- SQLAlchemy async with asyncpg (PostgreSQL)
- Use proper relationship loading strategies
- Close sessions via dependencies

---

### File Organization

```
frontend/src/
├── components/           # React components
│   └── ui/              # shadcn/ui components
├── pages/               # Page components (routes)
│   └── analysis/        # Feature folder (index, context, hooks, components)
├── hooks/               # Custom React hooks (useAuth, useWebSocket, etc.)
├── lib/                 # Utilities and API clients
├── contexts/            # React Context providers
├── types.ts             # Shared domain types & constants
└── const.ts             # App-level constants & OAuth helpers

backend/
├── api/                 # FastAPI application
│   ├── routers/         # Route handlers
│   ├── services/        # Business logic
│   ├── tests/           # API unit tests
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   ├── deps.py          # Dependencies
│   └── main.py          # App entry point
├── pipeline/            # CV processing
│   └── src/             # Pipeline modules
└── tests/               # Pipeline/integration tests
```

---

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root (see `env.example`):

```bash
LOCAL_DEV_MODE=true
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/football_dashboard
LOCAL_STORAGE_DIR=./uploads
OWNER_OPEN_ID=local-dev-user
```

### Running the Application

| Terminal | Command |
|----------|---------|
| 1 | `pnpm dev:api` (or `cd backend && uvicorn api.main:app --port 8000 --reload`) |
| 2 | `pnpm dev` (or `cd frontend && pnpm dev`) |

Access the app at **http://localhost:5173**

The CV pipeline worker runs separately on GPU infrastructure:
```bash
DASHBOARD_URL=http://localhost:8000 python backend/pipeline/worker.py
# or via Docker:
docker build -f docker/Dockerfile.worker -t football-worker .
docker run -e DASHBOARD_URL=https://your-endpoint football-worker
```

---

## Additional Notes

- Uses `wouter` for React routing (not React Router)
- Uses `sonner` for toast notifications
- Uses `recharts` for data visualization
- Uses `ruff` for Python linting
- Uses `vitest` + React Testing Library for frontend unit tests
- Uses `pytest` + `httpx` for backend API tests
- Uses `playwright` for E2E tests (run from repo root with `pnpm test:e2e`)
