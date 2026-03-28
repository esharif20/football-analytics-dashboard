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

test.beforeAll(async ({ request }) => {
  const res = await request.post(`${apiBase}/test-support/seed-analysis`);
  expect(res.ok(), 'seed-analysis must succeed — is ENABLE_TEST_SUPPORT=true set?').toBeTruthy();
});

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

test('dashboard loads without console errors', async ({ page }) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');

  // Wait for the page to render something meaningful
  await page.waitForSelector('h1, .card, [class*="grid"], [class*="container"]', { timeout: 10_000 });

  expect(
    consoleErrors,
    `Console errors found:\n${consoleErrors.join('\n')}`
  ).toHaveLength(0);
});

test('dashboard shows at least one analysis after seed', async ({ page }, testInfo) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');

  // Screenshot the full page state
  const screenshotPath = testInfo.outputPath('dashboard-analyses.png');
  await page.screenshot({ path: screenshotPath, fullPage: true });

  // Assert at least one analysis entry exists (completed badge or analysis link)
  const hasCompletedText = await page.locator('text=/completed/i').first().isVisible().catch(() => false);
  const hasAnalysisLink = await page.locator('a[href*="/analysis/"]').first().isVisible().catch(() => false);

  expect(
    hasCompletedText || hasAnalysisLink,
    'Expected at least one completed analysis to be visible after seeding. Dashboard may not be loading analyses.'
  ).toBe(true);
});

test('dashboard delete video shows confirmation and handles API error gracefully', async ({ page }, testInfo) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');

  // Look for a delete button on any video card
  const deleteButton = page.locator('button[aria-label*="delete" i], button[title*="delete" i]').first();
  const trashButton = page.locator('button').filter({ has: page.locator('svg') }).last();

  const hasDeleteButton = await deleteButton.isVisible().catch(() => false);

  if (!hasDeleteButton) {
    // Use the trash icon button as fallback
    const hasTrash = await trashButton.isVisible().catch(() => false);
    if (!hasTrash) {
      testInfo.annotations.push({
        type: 'info',
        description: 'No delete button found on dashboard — skipping delete error test (no videos rendered)',
      });
      return;
    }
  }

  // Mock delete endpoint to return 500
  await page.route('**/videos/**', route => {
    if (route.request().method() === 'DELETE') {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'simulated server error' }),
      });
    } else {
      route.continue();
    }
  });

  // Accept the confirm() dialog
  page.once('dialog', dialog => dialog.accept());

  // Click delete button
  if (hasDeleteButton) {
    await deleteButton.click();
  } else {
    await trashButton.click();
  }

  // Wait for toast error to appear
  const toastVisible = await page.locator('[data-sonner-toast], [role="status"]').filter({ hasText: /failed/i }).first()
    .waitFor({ state: 'visible', timeout: 5_000 })
    .then(() => true)
    .catch(() => false);

  const errorTextVisible = await page.locator('text=/failed/i').first().isVisible().catch(() => false);

  expect(
    toastVisible || errorTextVisible,
    'Expected a toast/error message containing "failed" after a 500 delete response. UI may be silently swallowing the error.'
  ).toBe(true);
});

test('navigation to upload page works', async ({ page }) => {
  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');

  // Find and click "Upload Video" or "New Analysis" button/link
  const uploadLink = page.locator('a[href="/upload"], button:has-text("Upload"), a:has-text("Upload")').first();
  await expect(uploadLink).toBeVisible({ timeout: 8_000 });
  await uploadLink.click();

  await page.waitForURL('**/upload', { timeout: 8_000 });
  expect(page.url()).toContain('/upload');

  // No console errors during navigation
  const navErrors = consoleErrors.filter(e => !e.includes('favicon'));
  expect(navErrors, `Console errors during navigation:\n${navErrors.join('\n')}`).toHaveLength(0);
});
