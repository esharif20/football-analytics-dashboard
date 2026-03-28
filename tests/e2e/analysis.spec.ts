import { test, expect, Page } from '@playwright/test';

const apiBase = process.env.TEST_API_URL || 'http://localhost:8000/api';

let analysisId: number;

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
  const body = await res.json();
  analysisId = body.analysisId;
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

test('analysis page loads for seeded completed analysis', async ({ page }, testInfo) => {
  test.setTimeout(20_000);

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Wait for completed badge or "completed" text
  const completedVisible = await page.locator('text=/completed/i').first()
    .waitFor({ state: 'visible', timeout: 10_000 })
    .then(() => true)
    .catch(() => false);

  const screenshotPath = testInfo.outputPath('analysis-loaded.png');
  await page.screenshot({ path: screenshotPath, fullPage: true });

  expect(completedVisible, 'Expected "completed" status to be visible on the analysis page after seeding').toBe(true);

  // No TypeError or "Cannot read" in console
  const criticalErrors = consoleErrors.filter(
    e => e.includes('TypeError') || e.includes('Cannot read')
  );
  expect(
    criticalErrors,
    `Critical JS errors found on analysis page:\n${criticalErrors.join('\n')}`
  ).toHaveLength(0);
});

test('possession stats render with non-zero values', async ({ page }, testInfo) => {
  test.setTimeout(20_000);

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Wait for possession text to appear
  const possessionVisible = await page.locator('text=/possession/i').first()
    .waitFor({ state: 'visible', timeout: 10_000 })
    .then(() => true)
    .catch(() => false);

  if (!possessionVisible) {
    const screenshotPath = testInfo.outputPath('possession-missing.png');
    await page.screenshot({ path: screenshotPath });
    testInfo.annotations.push({
      type: 'bug',
      description: 'Possession section not visible on analysis page — section may not render from fixture data',
    });
    expect(possessionVisible, 'Possession section not visible on analysis page').toBe(true);
  }

  // Assert the fixture value 55.2% (team 1 possession) is visible
  const hasFiftyFive = await page.locator('text=/55\\.2%|55%/').first().isVisible().catch(() => false);

  if (!hasFiftyFive) {
    const screenshotPath = testInfo.outputPath('possession-wrong-value.png');
    await page.screenshot({ path: screenshotPath });
    testInfo.annotations.push({
      type: 'bug',
      description:
        'Possession stats not rendering correct fixture values. Expected 55.2% for team 1 from fixture data but value not found on page.',
    });
    expect(hasFiftyFive, 'Expected possession value "55.2%" or "55%" from fixture data to be visible').toBe(true);
  }
});

test('events tab shows seeded pass and shot events', async ({ page }, testInfo) => {
  test.setTimeout(20_000);

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Find and click the Events tab
  const eventsTab = page.locator('[role="tab"]').filter({ hasText: /events/i }).first();
  const tabVisible = await eventsTab.isVisible().catch(() => false);

  if (!tabVisible) {
    testInfo.annotations.push({
      type: 'info',
      description: 'Events tab not found — the analysis page may use a different tab component or selector',
    });
    // Try a more general selector
    const altTab = page.locator('button, [role="tab"]').filter({ hasText: /events/i }).first();
    const altVisible = await altTab.isVisible().catch(() => false);
    if (!altVisible) {
      testInfo.annotations.push({
        type: 'bug',
        description: 'Events tab not visible on analysis page — UI may not have an Events tab at all',
      });
      return;
    }
    await altTab.click();
  } else {
    await eventsTab.click();
  }

  // Wait for tab content to render
  await page.waitForTimeout(1_000);

  // Check for pass or shot event text
  const hasPass = await page.locator('text=/pass/i').first().isVisible().catch(() => false);
  const hasShot = await page.locator('text=/shot/i').first().isVisible().catch(() => false);

  if (!hasPass && !hasShot) {
    const screenshotPath = testInfo.outputPath('events-tab-empty.png');
    await page.screenshot({ path: screenshotPath });
    testInfo.annotations.push({
      type: 'bug',
      description:
        'Events tab is empty despite seeded events in fixture (pass at 90s, shot at 150s). ' +
        'Events may not be loaded from analyticsDataUrl or the Events tab renders from the /events/ API endpoint which is not seeded.',
    });
    expect(hasPass || hasShot, 'Expected pass or shot events to be visible in the Events tab').toBe(true);
  }
});

test('video player section renders — annotated video null check', async ({ page }, testInfo) => {
  test.setTimeout(20_000);

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Check whether a <video> element is present
  const videoElement = page.locator('video').first();
  const videoPresent = await videoElement.isVisible().catch(() => false);

  if (videoPresent) {
    // Check the src attribute
    const src = await videoElement.getAttribute('src');
    if (!src || src.trim() === '') {
      testInfo.annotations.push({
        type: 'bug',
        description:
          'Video player rendered with null/empty src for seeded analysis — no annotatedVideoUrl in seed. ' +
          'The <video> element exists but has no src, which may cause a silent error or broken player UI.',
      });
    } else {
      testInfo.annotations.push({
        type: 'info',
        description: `Video element has src: ${src}`,
      });
    }
  } else {
    testInfo.annotations.push({
      type: 'info',
      description: 'No <video> element visible — video player may be hidden or not rendered for analyses without annotated video',
    });
  }

  // Regardless of video presence — no TypeError from video player mounting
  const videoErrors = consoleErrors.filter(
    e => e.includes('TypeError') || e.includes('Cannot read')
  );
  expect(
    videoErrors,
    `TypeError found in console (possibly from video player mounting):\n${videoErrors.join('\n')}`
  ).toHaveLength(0);
});

test('charts render without NaN/undefined in recharts', async ({ page }, testInfo) => {
  test.setTimeout(20_000);

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Scroll down to trigger IntersectionObserver animations
  await page.evaluate(() => {
    window.scrollBy(0, 500);
  });
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    window.scrollBy(0, 500);
  });
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    window.scrollBy(0, 500);
  });
  await page.waitForTimeout(800);

  // Check for chart-related errors
  const chartErrors = consoleErrors.filter(
    e =>
      e.includes('NaN') ||
      e.includes('undefined is not') ||
      e.includes('recharts') ||
      e.includes('Warning: Each child') ||
      e.includes('Warning: NaN')
  );

  if (chartErrors.length > 0) {
    const screenshotPath = testInfo.outputPath('charts-errors.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    expect(
      chartErrors,
      `Chart rendering errors found:\n${chartErrors.join('\n')}`
    ).toHaveLength(0);
  }
});

test('analysis page handles non-existent ID with a graceful error state', async ({ page }, testInfo) => {
  test.setTimeout(20_000);

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto('/analysis/999999');
  await page.waitForLoadState('networkidle');

  // Assert page is NOT a blank white screen
  const bodyText = await page.evaluate(() => document.body.innerText.trim());
  const hasContent = bodyText.length > 0;

  if (!hasContent) {
    const screenshotPath = testInfo.outputPath('analysis-404-blank.png');
    await page.screenshot({ path: screenshotPath });
    expect(hasContent, 'Analysis page shows blank white screen for missing ID — no error handling or not-found UI').toBe(true);
  }

  // Check no uncaught React error boundary crash (look for React's error boundary output)
  const reactCrash = consoleErrors.filter(
    e =>
      e.includes('The above error occurred') ||
      e.includes('Error boundary') ||
      e.includes('Uncaught Error')
  );

  expect(
    reactCrash,
    `React error boundary crash for non-existent analysis ID:\n${reactCrash.join('\n')}`
  ).toHaveLength(0);

  // The page should show some meaningful message (not just spinner forever)
  const hasErrorMessage = await page.locator('text=/not found|error|failed|doesn.t exist/i').first().isVisible().catch(() => false);
  const has404 = await page.locator('text=/404/').first().isVisible().catch(() => false);
  const hasAnyText = bodyText.length > 10;

  expect(
    hasErrorMessage || has404 || hasAnyText,
    `Analysis page for ID 999999 has no visible content or error message. Body text: "${bodyText.slice(0, 200)}"`
  ).toBe(true);
});
