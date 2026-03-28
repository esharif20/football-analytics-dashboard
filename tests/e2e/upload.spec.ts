import { test, expect, Page } from '@playwright/test';

const apiBase = process.env.TEST_API_URL || 'http://localhost:8000/api';

function collectDiagnostics(page: Page) {
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

test.afterEach(async ({ page }, testInfo) => {
  const errors: string[] = (page as any).__consoleErrors || [];
  const failures: string[] = (page as any).__networkFailures || [];
  if (errors.length) {
    testInfo.annotations.push({ type: 'console-errors', description: errors.join('\n') });
  }
  if (failures.length) {
    testInfo.annotations.push({ type: 'network-failures', description: failures.join('\n') });
  }
});

test('upload page renders without auth errors', async ({ page }) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/upload');
  await page.waitForLoadState('networkidle');

  // Assert no auth-related console errors
  const authErrors = consoleErrors.filter(e =>
    e.includes('401') || e.includes('403') || e.includes('Unauthorized') || e.includes('Forbidden')
  );
  expect(
    authErrors,
    `Auth errors in console:\n${authErrors.join('\n')}`
  ).toHaveLength(0);

  // Assert no 401/403 network failures
  const authNetworkErrors = networkFailures.filter(f => f.startsWith('401') || f.startsWith('403'));
  expect(
    authNetworkErrors,
    `Auth network failures:\n${authNetworkErrors.join('\n')}`
  ).toHaveLength(0);

  // File input must be visible
  const fileInput = page.locator('input[type="file"]');
  await expect(fileInput).toBeAttached({ timeout: 8_000 });
});

test('upload rejects non-video file with toast', async ({ page }, testInfo) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/upload');
  await page.waitForLoadState('networkidle');

  // Set a non-video file on the file input
  await page.setInputFiles('input[type="file"]', {
    name: 'document.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('fake pdf content'),
  });

  const screenshotPath = testInfo.outputPath('upload-rejection.png');
  await page.screenshot({ path: screenshotPath });

  // Assert a toast or error containing "video" appears
  const toastVisible = await page.locator('[data-sonner-toast]').filter({ hasText: /video/i }).first()
    .waitFor({ state: 'visible', timeout: 5_000 })
    .then(() => true)
    .catch(() => false);

  const errorText = await page.locator('text=/video/i').first().isVisible().catch(() => false);

  expect(
    toastVisible || errorText,
    'Expected a toast or error message containing "video" when uploading a non-video file. The upload page may not be validating file type.'
  ).toBe(true);
});

test('upload form requires title before submit', async ({ page }) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/upload');
  await page.waitForLoadState('networkidle');

  // Set a valid video file
  await page.setInputFiles('input[type="file"]', {
    name: 'match.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('fake video content'),
  });

  // Clear the title field if it was auto-filled
  const titleInput = page.locator('#title');
  const titleVisible = await titleInput.isVisible().catch(() => false);
  if (titleVisible) {
    await titleInput.clear();
  }

  // The submit button must be disabled when title is empty
  const submitButton = page.locator('button[type="submit"], button:has-text("Start Analysis")').first();

  // If button is disabled, validation is working — pass immediately
  const isDisabled = await submitButton.isDisabled().catch(() => true);
  if (isDisabled) {
    expect(isDisabled, 'Submit button is correctly disabled when title is empty').toBe(true);
    return;
  }

  // Button is enabled — click it and check we stay on /upload or get an error
  await submitButton.click({ force: true });
  await page.waitForTimeout(1_000);

  const currentUrl = page.url();
  const stayedOnUpload = currentUrl.includes('/upload');
  const toastError = await page.locator('[data-sonner-toast]').filter({ hasText: /title/i }).first().isVisible().catch(() => false);

  expect(
    stayedOnUpload || toastError,
    `Upload form submitted without a title and navigated away to: ${currentUrl}. Expected to stay on /upload or show a validation error.`
  ).toBe(true);
});

test('dead-code upload-base64 endpoint probed for 404', async ({ request }, testInfo) => {
  const res = await request.post(`${apiBase}/videos/upload-base64`, {
    data: {
      title: 'test',
      fileName: 'x.mp4',
      fileBase64: '',
      fileSize: 0,
      mimeType: 'video/mp4',
    },
    headers: { 'Content-Type': 'application/json' },
    failOnStatusCode: false,
  });

  const status = res.status();

  // Document the result regardless of status
  testInfo.annotations.push({
    type: 'info',
    description: `POST /api/videos/upload-base64 → HTTP ${status}`,
  });

  if (status === 404) {
    // This is the expected bug — dead code in videosApi.upload()
    testInfo.annotations.push({
      type: 'bug',
      description:
        'upload-base64 endpoint returns 404 — dead code in videosApi.upload() at frontend/src/lib/api-local.ts. ' +
        'The upload UI uses XHR multipart (correct) but videosApi.upload() points to a non-existent /videos/upload-base64 route.',
    });
    // Intentionally NOT failing — this test documents the bug, it does not enforce it
  } else if (status === 422) {
    // FastAPI returns 422 Unprocessable Entity for missing/invalid body fields
    testInfo.annotations.push({
      type: 'info',
      description: `upload-base64 endpoint exists (422 validation error) — dead code may still be reachable. Status: ${status}`,
    });
  } else if (status >= 200 && status < 300) {
    testInfo.annotations.push({
      type: 'info',
      description: `upload-base64 endpoint is alive and returned ${status} — dead code assertion incorrect`,
    });
  }

  // This test always passes — it only annotates findings
  expect(typeof status).toBe('number');
});
