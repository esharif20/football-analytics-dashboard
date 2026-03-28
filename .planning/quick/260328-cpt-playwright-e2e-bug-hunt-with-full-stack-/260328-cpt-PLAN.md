---
phase: quick
plan: 260328-cpt
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/start-test-stack.sh
  - tests/e2e/dashboard.spec.ts
  - tests/e2e/upload.spec.ts
  - tests/e2e/analysis.spec.ts
autonomous: true
requirements: []

must_haves:
  truths:
    - "Dashboard page loads without JS errors and lists analyses"
    - "Upload page renders all form fields and rejects non-video files"
    - "Analysis page renders stats, charts, and events tab for a seeded analysis"
    - "Navigation between pages works without full-page reload errors"
    - "API errors (4xx/5xx) produce visible UI feedback, not silent failures"
    - "All failed network requests and console errors are captured and reported"
  artifacts:
    - path: "scripts/start-test-stack.sh"
      provides: "One-command stack startup (DB + backend + frontend)"
    - path: "tests/e2e/dashboard.spec.ts"
      provides: "Dashboard bug-hunting tests with console/network capture"
    - path: "tests/e2e/upload.spec.ts"
      provides: "Upload page validation and error-path tests"
    - path: "tests/e2e/analysis.spec.ts"
      provides: "Analysis page full render and tab navigation tests"
  key_links:
    - from: "tests/e2e/*.spec.ts"
      to: "POST http://localhost:8000/api/test-support/seed-analysis"
      via: "request.post() in beforeAll — seeds deterministic analysis"
      pattern: "seed-analysis"
    - from: "scripts/start-test-stack.sh"
      to: "uvicorn api.main:app"
      via: "ENABLE_TEST_SUPPORT=true LOCAL_DEV_MODE=true env flags"
      pattern: "ENABLE_TEST_SUPPORT"
---

<objective>
Write Playwright E2E tests that act as a bug-hunting harness: capture every console error, failed network request, and broken UI state across the Dashboard, Upload, and Analysis pages. Run against a live full stack (Docker MySQL + FastAPI + Vite). Report all bugs found with screenshot evidence baked into test failure output.

Purpose: Surface real runtime bugs the codebase is known to have (silent API failures, auth guard inconsistencies, broken video playback, chart rendering crashes) before any fix work begins.
Output: Startup script + three spec files that produce annotated failure screenshots and a console-error log on every assertion failure.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md

<!-- Key contracts the executor needs -->
<interfaces>
<!-- Seed endpoint — POST http://localhost:8000/api/test-support/seed-analysis -->
<!-- Returns: { analysisId: number, videoId: number, userId: number } -->
<!-- Requires ENABLE_TEST_SUPPORT=true in backend env -->
<!-- Creates: User(openId=OWNER_OPEN_ID), Video, Analysis(status=completed, analytics=fixture) -->

<!-- Playwright config baseline (playwright.config.ts): -->
<!-- baseURL: http://localhost:5173 -->
<!-- testDir: ./tests/e2e -->
<!-- timeout: 30_000, screenshot: only-on-failure, trace: retain-on-failure -->

<!-- Auth: LOCAL_DEV_MODE=true → AutoLoginMiddleware logs every request in automatically -->
<!-- No login UI interaction needed — just hit any page and the session cookie is set -->

<!-- Routes (wouter): -->
<!--   /             → Home/landing -->
<!--   /dashboard    → Dashboard.tsx — lists videos and analyses -->
<!--   /upload       → Upload.tsx — multipart XHR upload -->
<!--   /analysis/:id → Analysis.tsx — 2000-line results page with tabs -->

<!-- Known issues to actively probe: -->
<!--   1. videosApi.upload() uses base64 but Upload.tsx uses XHR — dead code, possible 404 on /videos/upload-base64 -->
<!--   2. api-local.ts always sets Content-Type: application/json — would break FormData -->
<!--   3. No centralized auth guard — each page checks independently (race conditions possible) -->
<!--   4. Analysis.tsx is ~2000 lines — IntersectionObserver animations may hide content -->
<!--   5. annotatedVideoUrl may be null for seeded analysis (no actual video uploaded) -->
<!--   6. WebSocket errors silently swallowed in vite.config.ts -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write startup script for full-stack test environment</name>
  <files>scripts/start-test-stack.sh</files>
  <action>
Create an executable bash script at `scripts/start-test-stack.sh` that starts all three services needed before Playwright runs. The script must:

1. Start MySQL container in detached mode:
   ```bash
   docker compose up db -d
   ```
   Then wait for MySQL to be healthy by polling `docker compose ps db` until status shows "healthy" or by using `mysqladmin ping` loop (max 30s, 2s intervals).

2. Start FastAPI backend in background with test-support flags:
   ```bash
   cd backend && ENABLE_TEST_SUPPORT=true LOCAL_DEV_MODE=true \
     uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload \
     > /tmp/api-server.log 2>&1 &
   echo $! > /tmp/api-server.pid
   ```
   Then wait for backend readiness by polling `curl -sf http://localhost:8000/api/health` or `curl -sf http://localhost:8000/docs` (max 30s, 2s intervals). If it never responds, `cat /tmp/api-server.log` and exit 1.

3. Start Vite frontend in background:
   ```bash
   cd frontend && pnpm dev > /tmp/vite-server.log 2>&1 &
   echo $! > /tmp/vite-server.pid
   ```
   Wait for frontend readiness by polling `curl -sf http://localhost:5173` (max 30s, 2s intervals). If it never responds, `cat /tmp/vite-server.log` and exit 1.

4. Print a success banner with all three URLs and the command to stop servers:
   ```
   Stack ready:
     DB:       localhost:3307 (MySQL)
     Backend:  http://localhost:8000
     Frontend: http://localhost:5173

   Run tests: pnpm exec playwright test
   Stop servers: kill $(cat /tmp/api-server.pid) $(cat /tmp/vite-server.pid)
   ```

Mark the script executable: `chmod +x scripts/start-test-stack.sh`

The script must be idempotent — if a port is already in use, print a warning but do not fail (the stack may already be running).
  </action>
  <verify>bash scripts/start-test-stack.sh --help 2>&1 || bash -n scripts/start-test-stack.sh && echo "syntax ok"</verify>
  <done>Script exists at scripts/start-test-stack.sh, is executable, passes bash syntax check, and correctly sequences DB → backend (with ENABLE_TEST_SUPPORT=true) → frontend startup with health-check polling.</done>
</task>

<task type="auto">
  <name>Task 2: Write bug-hunting E2E specs for Dashboard, Upload, and Analysis pages</name>
  <files>
    tests/e2e/dashboard.spec.ts
    tests/e2e/upload.spec.ts
    tests/e2e/analysis.spec.ts
  </files>
  <action>
Write three spec files using the Playwright test API. Every spec MUST:
- Capture all browser console errors into a `consoleErrors: string[]` array
- Capture all failed network responses (status >= 400) into `networkFailures: string[]`
- On assertion failure, attach both arrays to the test report via `test.info().annotations`
- Take a manual screenshot on interesting negative states via `page.screenshot({ path: ... })`

Use this shared helper pattern at the top of each spec (or extract to `tests/e2e/helpers.ts`):

```typescript
async function collectDiagnostics(page: Page) {
  const consoleErrors: string[] = [];
  const networkFailures: string[] = [];
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('response', resp => {
    if (resp.status() >= 400) {
      networkFailures.push(`${resp.status()} ${resp.url()}`);
    }
  });
  return { consoleErrors, networkFailures };
}
```

Call `test.info().annotations.push(...)` in `afterEach` to attach arrays if non-empty.

---

**tests/e2e/dashboard.spec.ts** — Dashboard bug hunt

```typescript
const apiBase = process.env.TEST_API_URL || 'http://localhost:8000/api';

test.beforeAll(async ({ request }) => {
  // Seed so dashboard has at least one analysis to show
  const res = await request.post(`${apiBase}/test-support/seed-analysis`);
  expect(res.ok(), 'seed-analysis must succeed — is ENABLE_TEST_SUPPORT=true set?').toBeTruthy();
});
```

Tests to write:

1. **"dashboard loads without console errors"** — Navigate to `/dashboard`. Wait for `[data-testid="dashboard-content"]` OR the analyses list container to appear (use `page.waitForSelector` with a broad selector like `h1, .card, [class*="grid"]`). Assert `consoleErrors` is empty. If not, fail with the full list as the message.

2. **"dashboard shows at least one analysis after seed"** — Navigate to `/dashboard`. Wait for network idle. Assert that text matching `/completed/i` or a link containing `/analysis/` appears on the page. Screenshot the full page as `test-results/dashboard-analyses.png`.

3. **"dashboard delete video shows confirmation and handles API error gracefully"** — Navigate to `/dashboard`. If a delete button exists (`button[aria-label*="delete"], button:has(svg)` near a trash icon), mock the delete endpoint to return 500 using `page.route('**/videos/**', route => route.fulfill({ status: 500, body: JSON.stringify({ detail: 'simulated error' }) }))`. Click the delete button, dismiss/accept the confirm dialog via `page.on('dialog', d => d.accept())`. Assert a toast error appears (look for text matching `/failed/i` or `[data-sonner-toast]`).

4. **"navigation to upload page works"** — From `/dashboard`, click a link/button containing "Upload" or "New Analysis". Assert URL changes to `/upload`. No console errors.

---

**tests/e2e/upload.spec.ts** — Upload page bug hunt

Tests to write:

1. **"upload page renders without auth errors"** — Navigate to `/upload`. Assert no console errors containing `401` or `403` or `Unauthorized`. Assert the file input (`input[type="file"]`) is visible.

2. **"upload rejects non-video file with toast"** — Navigate to `/upload`. Use `page.setInputFiles('input[type="file"]', { name: 'document.pdf', mimeType: 'application/pdf', buffer: Buffer.from('fake pdf') })`. Assert a toast error appears (text matching `/video/i`). Screenshot as `test-results/upload-rejection.png`.

3. **"upload form requires title before submit"** — Navigate to `/upload`. Set a valid video file via `page.setInputFiles`: `{ name: 'match.mp4', mimeType: 'video/mp4', buffer: Buffer.from('fake video') }`. Clear the title field if auto-filled. Click the submit button. Assert that form does not navigate away (URL still `/upload`) OR that a validation error appears.

4. **"dead-code upload-base64 endpoint probed for 404"** — Make a direct API request to `POST http://localhost:8000/api/videos/upload-base64` with a minimal JSON body `{ title: 'test', fileName: 'x.mp4', fileBase64: '', fileSize: 0, mimeType: 'video/mp4' }`. Log the response status as a test annotation. If the response is 404, mark it as a known bug annotation: `test.info().annotations.push({ type: 'bug', description: 'upload-base64 endpoint returns 404 — dead code in videosApi.upload()' })`. The test should NOT fail on 404 — it documents the bug.

---

**tests/e2e/analysis.spec.ts** — Analysis page bug hunt

```typescript
let analysisId: number;

test.beforeAll(async ({ request }) => {
  const res = await request.post(`${apiBase}/test-support/seed-analysis`);
  expect(res.ok(), 'seed-analysis must succeed').toBeTruthy();
  const body = await res.json();
  analysisId = body.analysisId;
});
```

Tests to write:

1. **"analysis page loads for seeded completed analysis"** — Navigate to `/analysis/${analysisId}`. Wait for the page to settle (wait for `.completed` badge OR text matching `/completed/i` with timeout 10s). Screenshot as `test-results/analysis-loaded.png`. Assert no console errors containing `TypeError` or `Cannot read`.

2. **"possession stats render with non-zero values"** — Navigate to `/analysis/${analysisId}`. Wait for text matching `/possession/i` to appear. Assert that text matching `/55\.2%|55%/` is visible (fixture value for team 1). If it is NOT visible, screenshot and annotate as bug: "Possession stats not rendering from fixture data".

3. **"events tab shows seeded pass and shot events"** — Navigate to `/analysis/${analysisId}`. Find and click the "Events" tab (look for `[role="tab"]` with text matching `/events/i`). Wait for the tab content to appear. Assert text matching `/pass/i` or `/shot/i` is visible. If the tab content is empty, screenshot and annotate as bug: "Events tab empty despite seeded events in fixture".

4. **"video player section renders — annotated video null check"** — Navigate to `/analysis/${analysisId}`. Check whether a `<video>` element is present. If present, check `src` attribute — if `src` is empty or null, annotate as bug: "Video player rendered with null/empty src for seeded analysis (no annotated video in seed)". This is an expected bug — the test documents it, not fails on it. Assert no `TypeError` in console from the video player mounting.

5. **"charts render without NaN/undefined in recharts"** — Navigate to `/analysis/${analysisId}`. Scroll down to trigger IntersectionObserver animations. Check `consoleErrors` for any message containing `NaN`, `undefined is not`, `recharts`, or `Warning: Each child`. If found, fail with the full error list as message and screenshot.

6. **"analysis page handles non-existent ID with a graceful error state"** — Navigate to `/analysis/999999`. Assert page does NOT show a blank white screen (check that some visible text exists on the page). Assert no uncaught React error boundary crash. If page is blank, screenshot as `test-results/analysis-404-blank.png` and fail: "Analysis page shows blank for missing ID — no error handling".

---

**Implementation notes for the executor:**

- Import `Page` from `@playwright/test` for the helper type annotation.
- Use `page.waitForLoadState('networkidle')` after navigation where content is data-fetched.
- The `afterEach` hook should attach diagnostics even if the test passed:
  ```typescript
  test.afterEach(async ({ page }, testInfo) => {
    const errors = (page as any).__consoleErrors || [];
    const failures = (page as any).__networkFailures || [];
    if (errors.length) testInfo.annotations.push({ type: 'console-errors', description: errors.join('\n') });
    if (failures.length) testInfo.annotations.push({ type: 'network-failures', description: failures.join('\n') });
  });
  ```
  Because `collectDiagnostics` returns closures over arrays, store refs on the page object or in the test scope via `test.extend` fixture if preferred.
- Set `test.setTimeout(20_000)` for analysis page tests due to React Query data fetching.
- The `screenshot` calls use relative paths from the project root — ensure `test-results/` directory exists or use `testInfo.outputPath('filename.png')` which is always valid.
  </action>
  <verify>cd /Users/eshansharif/Documents/football-analytics-dashboard && pnpm exec tsc --noEmit --project frontend/tsconfig.json 2>&1 | grep -E "tests/e2e" || echo "no TS errors in specs (or tsconfig excludes tests)"</verify>
  <done>
    - tests/e2e/dashboard.spec.ts exists with 4 tests covering load, listing, delete error, and navigation
    - tests/e2e/upload.spec.ts exists with 4 tests covering auth, rejection, validation, and dead-code endpoint probe
    - tests/e2e/analysis.spec.ts exists with 6 tests covering load, stats, events, video null, charts, and 404 handling
    - Every test captures console errors and network failures
    - Bug-documenting tests use test.info().annotations (not hard failures) for known issues
    - tests that CAN fail hard DO fail hard (TypeError in console, blank 404 page, missing possession stats)
  </done>
</task>

<task type="auto">
  <name>Task 3: Run the full test suite and produce a bug report</name>
  <files>test-results/BUG-REPORT.md</files>
  <action>
With the servers already running (via `scripts/start-test-stack.sh` or manually), run the full Playwright suite and capture results. Then generate a structured bug report.

**Step 1 — Start servers if not running:**

Check whether servers are up before running tests:
```bash
curl -sf http://localhost:8000/docs > /dev/null || bash scripts/start-test-stack.sh
curl -sf http://localhost:5173 > /dev/null || echo "Vite not running — start manually"
```

Wait up to 15 seconds for both to respond.

**Step 2 — Run the full Playwright suite:**

```bash
cd /Users/eshansharif/Documents/football-analytics-dashboard
pnpm exec playwright test --reporter=list,json 2>&1 | tee /tmp/playwright-run.log
```

Capture the JSON reporter output (written to `playwright-report/results.json` by default). If the JSON file does not appear, re-run with explicit output path:
```bash
pnpm exec playwright test --reporter=json:test-results/results.json 2>&1 | tee /tmp/playwright-run.log
```

**Step 3 — Parse results and write BUG-REPORT.md:**

Read `/tmp/playwright-run.log` and `test-results/results.json` (if it exists). Write `test-results/BUG-REPORT.md` with the following structure:

```markdown
# E2E Bug Report — Football Analytics Dashboard
Generated: {ISO timestamp}
Stack: FastAPI :8000 + Vite :5173 + MySQL :3307

## Summary

| Metric | Count |
|--------|-------|
| Tests run | N |
| Passed | N |
| Failed | N |
| Bug annotations | N |

## Confirmed Bugs (Hard Failures)

For each FAILED test:

### BUG-{N}: {test title}
**Spec:** {file}:{line}
**Failure:** {error message}
**Screenshot:** {path if exists}
**Console errors captured:** {list if any}
**Network failures captured:** {list if any}

## Documented Issues (Annotation-only)

For each test with `type: "bug"` annotation:

### ISSUE-{N}: {annotation description}
**Found in test:** {test title}
**Severity:** Needs investigation

## Console Errors Captured

Deduplicated list of all console errors seen across tests.

## Network Failures Captured

Deduplicated list of all HTTP 4xx/5xx seen across tests.

## Screenshots

List all PNG files in test-results/ with their associated test.
```

If `test-results/results.json` is not available, parse the plain text log from `/tmp/playwright-run.log` to extract PASSED/FAILED test names and write what is available.

**Step 4 — Print the summary to stdout** so it appears in the Claude execution log.

This task is the "ship" step — its output is the deliverable. A successful run means the bug report exists even if tests fail (bugs found = mission success).
  </action>
  <verify>test -f test-results/BUG-REPORT.md && echo "BUG-REPORT.md exists" && head -20 test-results/BUG-REPORT.md</verify>
  <done>
    - test-results/BUG-REPORT.md exists and contains a Summary table
    - All confirmed bugs (hard test failures) are listed with failure message and screenshot path
    - All annotation-only issues are listed separately
    - Console errors and network failures sections are populated (or explicitly state "none captured")
    - The report is readable without running any tools — it is the final deliverable
  </done>
</task>

</tasks>

<verification>
After all tasks complete:
1. `ls tests/e2e/` shows dashboard.spec.ts, upload.spec.ts, analysis.spec.ts (plus existing commentary.spec.ts)
2. `bash -n scripts/start-test-stack.sh` passes (no syntax errors)
3. `test-results/BUG-REPORT.md` exists and has a Summary section
4. At least one screenshot exists in `test-results/` for the analysis page load test
5. The BUG-REPORT.md distinguishes between hard failures (real bugs) and annotation issues (known/expected)
</verification>

<success_criteria>
- Three spec files exist covering all four page routes (home via dashboard, upload, analysis)
- Every test captures console errors and network failures — no test is "silent" on error
- Bug-documenting tests (base64 endpoint probe, video null src) use annotations not hard failures
- Tests that should catch real bugs DO fail hard (blank 404 page, TypeError in console, empty stats)
- Startup script handles the full DB → backend → frontend sequence with health-check polling
- BUG-REPORT.md is produced after the run, listing all findings with screenshot references
</success_criteria>

<output>
After completion, create `.planning/quick/260328-cpt-playwright-e2e-bug-hunt-with-full-stack-/260328-cpt-SUMMARY.md` with what was built, what bugs were found, and links to the spec files.
</output>
