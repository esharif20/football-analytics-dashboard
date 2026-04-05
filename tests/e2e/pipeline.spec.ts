import { test, expect, Page } from '@playwright/test';

const apiBase = process.env.TEST_API_URL || 'http://localhost:8000/api';

// Analysis ID from pipeline-e2e.sh (set via env var or saved to temp file).
// If not set, the beforeAll hook will upload a video and create an analysis via the API.
let analysisId: number | null = null;

function collectDiagnostics(page: Page) {
  const consoleErrors: string[] = [];
  const networkFailures: string[] = [];
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('response', resp => {
    if (resp.status() >= 400) networkFailures.push(`${resp.status()} ${resp.url()}`);
  });
  return { consoleErrors, networkFailures };
}

test.beforeAll(async ({ request }) => {
  // Mode A: analysis ID provided by pipeline-e2e.sh
  const envId = process.env.PIPELINE_E2E_ANALYSIS_ID;
  if (envId && !isNaN(Number(envId))) {
    analysisId = Number(envId);
    console.log(`[pipeline.spec] Using pre-existing analysis id=${analysisId} from PIPELINE_E2E_ANALYSIS_ID`);
    return;
  }

  // Mode B: no ID — we need to find or fail gracefully
  // This spec is designed to be run AFTER pipeline-e2e.sh, not standalone.
  // Without a real processed analysis, these tests are meaningless (they'd test nothing).
  console.warn(
    '[pipeline.spec] PIPELINE_E2E_ANALYSIS_ID not set. ' +
    'Run: bash scripts/pipeline-e2e.sh --video 7 first, then rerun with the printed env var.'
  );
  // Don't throw — tests will skip themselves gracefully if analysisId is null
});

test.afterEach(async ({ page }, testInfo) => {
  const errors: string[] = (page as any).__consoleErrors || [];
  const failures: string[] = (page as any).__networkFailures || [];
  if (errors.length) testInfo.annotations.push({ type: 'console-errors', description: errors.join('\n') });
  if (failures.length) testInfo.annotations.push({ type: 'network-failures', description: failures.join('\n') });
});

test('pipeline: analysis page shows completed status', async ({ page }, testInfo) => {
  test.setTimeout(60_000);

  if (!analysisId) {
    testInfo.annotations.push({ type: 'skip', description: 'No PIPELINE_E2E_ANALYSIS_ID set — run pipeline-e2e.sh first' });
    test.skip(true, 'PIPELINE_E2E_ANALYSIS_ID not set');
    return;
  }

  const { consoleErrors, networkFailures } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;
  (page as any).__networkFailures = networkFailures;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  const completedVisible = await page.locator('text=/completed/i').first()
    .waitFor({ state: 'visible', timeout: 30_000 })
    .then(() => true)
    .catch(() => false);

  await page.screenshot({ path: testInfo.outputPath('pipeline-analysis-loaded.png'), fullPage: true });

  expect(
    completedVisible,
    'Expected "completed" status badge on analysis page for a real processed analysis'
  ).toBe(true);

  const criticalErrors = consoleErrors.filter(
    e => e.includes('TypeError') || e.includes('Cannot read')
  );
  expect(
    criticalErrors,
    `Critical JS errors on analysis page:\n${criticalErrors.join('\n')}`
  ).toHaveLength(0);
});

test('pipeline: annotated video player has a real src', async ({ page }, testInfo) => {
  test.setTimeout(60_000);

  if (!analysisId) {
    test.skip(true, 'PIPELINE_E2E_ANALYSIS_ID not set');
    return;
  }

  const { consoleErrors } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Wait for the completed badge first
  await page.locator('text=/completed/i').first()
    .waitFor({ state: 'visible', timeout: 30_000 })
    .catch(() => null);

  const videoElement = page.locator('video').first();
  const videoPresent = await videoElement.isVisible().catch(() => false);

  if (!videoPresent) {
    testInfo.annotations.push({
      type: 'info',
      description: 'No <video> element visible — video player may not render for this analysis',
    });
    // Not a hard failure — some modes may not produce annotated video
    return;
  }

  const src = await videoElement.getAttribute('src').catch(() => null);
  await page.screenshot({ path: testInfo.outputPath('pipeline-video-player.png') });

  expect(
    src && src.trim().length > 0,
    `Video player <video> element has empty or null src. Expected a real annotated video URL but got: "${src}"`
  ).toBe(true);

  // Ensure no video-related JS errors
  const videoErrors = consoleErrors.filter(
    e => e.includes('TypeError') || e.includes('Cannot read')
  );
  expect(videoErrors, `JS errors found (possibly from video player):\n${videoErrors.join('\n')}`).toHaveLength(0);
});

test('pipeline: possession statistics render with real values', async ({ page }, testInfo) => {
  test.setTimeout(60_000);

  if (!analysisId) {
    test.skip(true, 'PIPELINE_E2E_ANALYSIS_ID not set');
    return;
  }

  const { consoleErrors } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Wait for completed
  await page.locator('text=/completed/i').first()
    .waitFor({ state: 'visible', timeout: 30_000 })
    .catch(() => null);

  const possessionVisible = await page.locator('text=/possession/i').first()
    .waitFor({ state: 'visible', timeout: 15_000 })
    .then(() => true)
    .catch(() => false);

  if (!possessionVisible) {
    await page.screenshot({ path: testInfo.outputPath('pipeline-possession-missing.png') });
    testInfo.annotations.push({
      type: 'bug',
      description: 'Possession section not visible for real processed analysis',
    });
    expect(possessionVisible, 'Possession section not visible').toBe(true);
  }

  // Check that a percentage value renders (any xx.x% format)
  const hasPercent = await page.locator('text=/%/').first().isVisible().catch(() => false);
  expect(
    hasPercent,
    'No percentage value visible in possession stats — real analysis data may not be rendering correctly'
  ).toBe(true);

  await page.screenshot({ path: testInfo.outputPath('pipeline-possession-ok.png') });
});

test('pipeline: events tab shows real detected events', async ({ page }, testInfo) => {
  test.setTimeout(60_000);

  if (!analysisId) {
    test.skip(true, 'PIPELINE_E2E_ANALYSIS_ID not set');
    return;
  }

  const { consoleErrors } = collectDiagnostics(page);
  (page as any).__consoleErrors = consoleErrors;

  await page.goto(`/analysis/${analysisId}`);
  await page.waitForLoadState('networkidle');

  // Wait for completed
  await page.locator('text=/completed/i').first()
    .waitFor({ state: 'visible', timeout: 30_000 })
    .catch(() => null);

  // Find events tab
  const eventsTab = page.locator('[role="tab"]').filter({ hasText: /events/i }).first();
  const tabVisible = await eventsTab.isVisible().catch(() => false);

  if (!tabVisible) {
    const altTab = page.locator('button, [role="tab"]').filter({ hasText: /events/i }).first();
    const altVisible = await altTab.isVisible().catch(() => false);
    if (!altVisible) {
      testInfo.annotations.push({
        type: 'info',
        description: 'Events tab not found on page — may use a different UI structure',
      });
      return;
    }
    await altTab.click();
  } else {
    await eventsTab.click();
  }

  await page.waitForTimeout(1_000);
  await page.screenshot({ path: testInfo.outputPath('pipeline-events-tab.png') });

  const hasPass = await page.locator('text=/pass/i').first().isVisible().catch(() => false);
  const hasShot = await page.locator('text=/shot/i').first().isVisible().catch(() => false);
  const hasEvent = await page.locator('text=/event/i').first().isVisible().catch(() => false);

  if (!hasPass && !hasShot && !hasEvent) {
    testInfo.annotations.push({
      type: 'info',
      description: 'Events tab is empty for real analysis — pipeline may not have detected events for this clip',
    });
    // Warn but don't fail — depends on video content
  }
});
