import { test } from '@playwright/test';
import path from 'path';

const VIDEO = path.join(process.cwd(), 'test_uploads', 'Test1.mp4');

test('upload Test1.mp4 via UI', async ({ page }) => {
  await page.goto('/upload');
  await page.waitForLoadState('networkidle');

  // Wait for file input (hidden with id="video-upload")
  await page.locator('#video-upload').waitFor({ state: 'attached', timeout: 10000 });

  // Set file directly on hidden input
  await page.locator('#video-upload').setInputFiles(VIDEO);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '/tmp/after_file.png' });

  // Fill title if empty
  const titleInput = page.locator('#title').first();
  if (await titleInput.isVisible().catch(() => false)) {
    const val = await titleInput.inputValue().catch(() => '');
    if (!val) await titleInput.fill('Test1 - GPU Pipeline Test');
  }

  const submitBtn = page.locator('button[type="submit"], button:has-text("Start Analysis"), button:has-text("Analyse")').first();
  console.log('Submit button:', await submitBtn.textContent());
  await submitBtn.click();

  await page.waitForURL(/\/analysis\/\d+/, { timeout: 60000 });
  const url = page.url();
  const id = url.match(/\/analysis\/(\d+)/)?.[1];
  console.log('SUCCESS - Analysis ID:', id, 'URL:', url);
  await page.screenshot({ path: '/tmp/after_upload.png' });
});
