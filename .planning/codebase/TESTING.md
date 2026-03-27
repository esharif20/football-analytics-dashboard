---
focus: testing
source: manual-draft
---

# Testing

- No automated test suite configured (frontend or backend) per AGENTS.md; some placeholder test folders exist but no runner wired.
- Suggestions: Vitest + React Testing Library for frontend; pytest + httpx for FastAPI async routes; Playwright/E2E if needed.
- Current commands: none. TypeScript type-check via `pnpm check` once configured. Backend tests would require MySQL test DB + fixtures.
