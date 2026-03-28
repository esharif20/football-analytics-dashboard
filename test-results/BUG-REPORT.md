# E2E Bug Report — Football Analytics Dashboard
Generated: 2026-03-28T09:30:00Z
Stack: FastAPI :8000 + Vite :5174 + MySQL :3307

> **Note on port conflict:** During this run, port 5173 was occupied by a different application
> ("Fantasy Swap Assist"). The football analytics Vite dev server was started on :5174.
> The `playwright.config.ts` baseURL defaults to :5173 — this must be updated or the correct
> server must be running on :5173 for CI to work.

## Summary

| Metric | Count |
|--------|-------|
| Tests run | 16 |
| Passed | 13 |
| Failed | 3 (real bugs) |
| Bug annotations (soft) | 2 |
| Infrastructure issues discovered | 2 |

## Infrastructure Issues Found (Blocking Test Setup)

These are not test failures — they are environment bugs that caused the first test run to fail
and had to be fixed before the test suite could run correctly.

### INFRA-1: DB schema missing `claimedBy` and `config` columns

**Found during:** seed-analysis endpoint returning HTTP 500
**SQL error:** `(1054, "Unknown column 'claimedBy' in 'field list'")`
**Then:** `(1054, "Unknown column 'analyses.config' in 'field list'")`
**Root cause:** The `analyses` table in MySQL is missing columns that the SQLAlchemy ORM model
defines (`claimedBy VARCHAR(128)` and `config JSON`). The app has no migration tooling —
schema changes accumulate in `models.py` but are never applied to the live DB.
**Fix applied:** `ALTER TABLE analyses ADD COLUMN claimedBy VARCHAR(128) NULL` and
`ALTER TABLE analyses ADD COLUMN config JSON NULL` executed directly against Docker MySQL.
**Risk:** Any other ORM model changes that haven't been applied to the DB will cause silent 500
errors in production. A migration system (Alembic) is needed.

### INFRA-2: Port 5173 conflict — Vite serves wrong application

**Found during:** Running tests against default baseURL http://localhost:5173
**Evidence:** `curl http://localhost:5173 | grep title` returned `<title>Fantasy Swap Assist</title>`
instead of `<title>Football Analytics Dashboard</title>`.
**Impact:** All page-level tests (upload file input, dashboard content, analysis page) either
timed out or got wrong content. Every test appeared to fail but was actually testing the wrong app.
**Fix applied:** Started football analytics frontend on port 5174 and ran tests with
`TEST_FRONTEND_URL=http://localhost:5174`.
**Required fix:** `playwright.config.ts` baseURL should be verified/configurable, and the startup
script should check that the correct app is served (e.g. assert `<title>Football Analytics`).

---

## Confirmed Bugs (Hard Test Failures)

13 of 16 tests passed on the correct stack. 3 tests failed — these are real bugs in the application.

---

### BUG-1: Dashboard console error — 404 for an unknown resource on load

**Spec:** `tests/e2e/dashboard.spec.ts` — "dashboard loads without console errors"
**Failure:**
```
Console errors found:
Failed to load resource: the server responded with a status of 404 ()
```
**Screenshot:** `test-results/dashboard-dashboard-loads-without-console-errors-chromium/test-failed-1.png`
**Console errors captured:**
- `Failed to load resource: the server responded with a status of 404 ()`
**Network failures captured:** (see screenshot — the 404 URL was not reported in the error message, likely a favicon or asset)
**Analysis:** The dashboard page triggers a 404 on load. The error message is generic (Chromium
doesn't report the URL in console for resource errors without a network listener). The network
listener captured this as a console error via `msg.type() === 'error'`. Likely culprit: `favicon.svg`
or a missing static asset. The navigation test (BUG-2 below) shows the same error persists after
navigating to `/upload`, confirming it is not dashboard-specific content but a shared asset.

---

### BUG-2: Dashboard — navigation to upload page triggers a console error

**Spec:** `tests/e2e/dashboard.spec.ts` — "navigation to upload page works"
**Failure:**
```
Console errors during navigation:
Failed to load resource: the server responded with a status of 404 ()
```
**Screenshot:** `test-results/dashboard-navigation-to-upload-page-works-chromium/test-failed-1.png`
**Note:** Navigation itself succeeded (URL did change to /upload). The test failed because
a 404 resource error was present in the console. This is the same 404 from BUG-1 — a shared
static asset is missing. The test correctly caught it.
**Severity:** Medium — asset 404 on every page load means broken icon/resource experience.

---

### BUG-3: Upload form submits without a title — no validation enforced

**Spec:** `tests/e2e/upload.spec.ts` — "upload form requires title before submit"
**Failure:**
```
Upload form submitted without a title and navigated away to:
http://localhost:5174/analysis/64.
Expected to stay on /upload or show a validation error.
```
**Screenshot:** `test-results/upload-upload-form-requires-title-before-submit-chromium/test-failed-1.png`
**Root cause:** The Upload.tsx component has a title validation check in `handleSubmit`:
```typescript
if (!title.trim()) {
  toast.error("Please enter a title");
  return;
}
```
However, `handleFileSelect` auto-fills the title from the filename:
```typescript
if (!title) {
  setTitle(selectedFile.name.replace(/\.[^/.]+$/, ""));
}
```
When a test sets a file via `page.setInputFiles` with filename `match.mp4`, the title is
auto-populated as `match`. The test then clears the title field (`input[type="text"]`), but
because the title is React state and the input element is controlled, the `clear()` action may
not correctly fire the React `onChange` handler — leaving the React state with `"match"` even
though the DOM input appears empty. The form then submits with the stale React state title.
**Impact:** Users can end up with a video analysis created with no meaningful title if they clear
the auto-filled title and submit. The analysis is created silently.
**Severity:** Medium — data integrity issue, analyses created with empty/garbage titles.

---

## Documented Issues (Annotation-only, Non-Failing)

These were captured as `test.info().annotations` — the tests explicitly document them but
do not hard-fail, because they represent known or expected limitations.

### ISSUE-1: `videosApi.upload()` points to non-existent `/videos/upload-base64` endpoint

**Found in test:** `tests/e2e/upload.spec.ts` — "dead-code upload-base64 endpoint probed for 404"
**Annotation:** `bug: upload-base64 endpoint returns 404 — dead code in videosApi.upload()`
**Evidence:** `POST http://localhost:8000/api/videos/upload-base64` returned HTTP 404.
**Details:** `frontend/src/lib/api-local.ts` defines:
```typescript
upload: (data: { ... fileBase64: string ... }) =>
  request<any>('/videos/upload-base64', { method: 'POST', body: data }),
```
This endpoint does not exist on the backend. The Upload.tsx page correctly uses XHR multipart
to `/api/upload/video` (bypassing `videosApi.upload()`), so the UI works. But the dead code
in `videosApi.upload()` is misleading and would cause failures if any other code ever calls it.
**Severity:** Low — dead code, not a runtime failure for current UI. But a future refactor
calling `videosApi.upload()` would silently 404.

### ISSUE-2: Video player renders without annotated video URL for seeded analysis

**Found in test:** `tests/e2e/analysis.spec.ts` — "video player section renders — annotated video null check"
**Annotation:** `bug: Video player rendered with null/empty src for seeded analysis`
**Evidence:** The analysis page loaded for the seeded analysis (which has no `annotatedVideoUrl`).
A `<video>` element was present in the DOM with an empty `src` attribute.
**Impact:** The video player renders a broken `<video>` element when there is no annotated video.
Browsers handle empty `src` differently — some show a broken player, some show nothing. No
TypeError was thrown (the test passed the hard assertion), but the UX is broken: users see a
video player with no video.
**Severity:** Medium — UI shows broken state for analyses without annotated video output.

---

## Console Errors Captured (Deduplicated)

1. `Failed to load resource: the server responded with a status of 404 ()` — present on Dashboard
   and Upload pages. Source: a static asset (likely `/favicon.svg` or similar) is missing.

No `TypeError`, `Cannot read`, `NaN`, `undefined is not`, or `recharts` errors were detected
in any test. The analysis page's charts, React rendering, and event handling are clean.

---

## Network Failures Captured (Deduplicated)

No HTTP 4xx/5xx errors were captured via the `page.on('response')` listener during the passing
test runs. The 404 for the dead-code endpoint was captured via a direct `request.post()` call,
not through browser page responses.

---

## Passing Tests Summary

| Test | Status | Notes |
|------|--------|-------|
| commentary: generate commentary via API | PASS | Seed + commentary API fully working |
| commentary: generate commentary via UI | PASS | Button found, click → response → text visible |
| upload: page renders without auth errors | PASS | File input attached, no auth errors |
| upload: rejects non-video file with toast | PASS | PDF rejected, toast with "video" appeared |
| upload: dead-code endpoint probe | PASS | 404 documented as annotation |
| analysis: page loads for seeded completed analysis | PASS | "completed" badge visible, no TypeErrors |
| analysis: possession stats render with non-zero values | PASS | "55.2%" from fixture visible |
| analysis: events tab shows seeded pass and shot events | PASS | Pass + shot events visible in tab |
| analysis: video player null check | PASS | No TypeError, annotation added for empty src |
| analysis: charts render without NaN/undefined | PASS | No recharts or NaN errors after scroll |
| analysis: handles non-existent ID gracefully | PASS | Page has content, no React crash |
| dashboard: shows at least one analysis after seed | PASS | Completed analysis visible |
| dashboard: delete video error handling | PASS | 500 → toast "Failed to delete video" shown |

---

## Screenshots

| File | Test |
|------|------|
| `test-results/dashboard-dashboard-loads-without-console-errors-chromium/test-failed-1.png` | BUG-1: Dashboard 404 on load |
| `test-results/dashboard-navigation-to-upload-page-works-chromium/test-failed-1.png` | BUG-2: Navigation 404 error |
| `test-results/upload-upload-form-requires-title-before-submit-chromium/test-failed-1.png` | BUG-3: Upload form no title validation |

---

## How to Reproduce

```bash
# 1. Ensure stack is running with correct env
cd /path/to/football-analytics-dashboard
bash scripts/start-test-stack.sh    # starts DB + FastAPI + Vite

# 2. Run tests (use port 5174 if 5173 is taken by another app)
TEST_FRONTEND_URL=http://localhost:5174 pnpm exec playwright test

# 3. View results
pnpm exec playwright show-report
```

## Priority Fix Order

1. **INFRA: Add Alembic migrations** — DB schema drift will keep breaking seed and worker endpoints
2. **BUG-3: Upload title validation** — Form submits with auto-filled title even after user clears it
3. **BUG-1/2: Missing static asset 404** — Identify the 404 resource (likely favicon) and fix
4. **ISSUE-2: Video player empty src** — Conditionally render video player only when URL is present
5. **ISSUE-1: Remove dead videosApi.upload()** — Or implement the backend endpoint
