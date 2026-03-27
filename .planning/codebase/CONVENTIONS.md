---
focus: conventions
source: manual-draft
---

# Conventions

- TypeScript strict; prefer path aliases `@/*`; components PascalCase, hooks `use*` prefix, utilities kebab-case.
- UI styling via Tailwind utilities + `cn()` helper; shadcn/ui primitives under `components/ui`.
- Data fetching: React Query for server state; WebSocket hook for progress streaming.
- Backend imports ordered stdlib → third-party → local; async SQLAlchemy session dependencies; Pydantic schemas for request/response DTOs.
- Constants UPPER_SNAKE_CASE; environment loaded via `config.py`; avoid `any` in TS and prefer explicit types.
- Package managers: pnpm (frontend) and pip (backend/pipeline); MySQL via docker-compose.
