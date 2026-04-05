import { expect, test } from '@playwright/test';

const apiBase = process.env.TEST_API_URL || 'http://localhost:8000/api';

let analysisId: number;

test.beforeAll(async ({ request }) => {
  const res = await request.post(`${apiBase}/test-support/seed-analysis`);
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  analysisId = body.analysisId;
});

test('generate commentary via API (stub provider)', async ({ request }) => {
  const res = await request.post(`${apiBase}/commentary/${analysisId}`, {
    data: { type: 'match_summary' },
  });

  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  expect(body.content).toContain('STUB-COMMENTARY');
  expect(body.type ?? body.analysis_type ?? 'match_summary').toBeDefined();

  const listRes = await request.get(`${apiBase}/commentary/${analysisId}`);
  expect(listRes.ok()).toBeTruthy();
  const list = await listRes.json();
  expect(Array.isArray(list)).toBe(true);
  expect(list.some((c: any) => (c.content || '').includes('STUB-COMMENTARY'))).toBe(true);
});

test('generate commentary via UI (stub provider)', async ({ page }) => {
  await page.goto(`/analysis/${analysisId}`);

  const generateButton = page.getByRole('button', { name: /Generate Tactical Analysis/i });
  await expect(generateButton).toBeVisible();

  const [response] = await Promise.all([
    page.waitForResponse((resp) => resp.url().includes(`/commentary/`) && resp.status() === 200),
    generateButton.click(),
  ]);

  expect(response.ok()).toBeTruthy();

  await expect(page.getByText('STUB-COMMENTARY')).toBeVisible();
});
