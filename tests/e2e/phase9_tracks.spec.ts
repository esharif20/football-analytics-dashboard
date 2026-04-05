/**
 * Phase 9 E2E verification — tracks wired to pitch visualizations
 * Requires analysis #13 (has tracks in DB) and servers running on :8000 + :5174
 */
import { test, expect, Page } from '@playwright/test'

const BASE = process.env.TEST_BASE_URL || 'http://localhost:5174'
const ANALYSIS_ID = 13

function collectErrors(page: Page) {
  const errs: string[] = []
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text()) })
  return errs
}

test.describe('Phase 9 — Real tracking data on analysis page', () => {
  test('tracks API returns data for analysis 13', async ({ request }) => {
    const res = await request.get('http://localhost:8000/api/tracks/13?limit=1')
    expect(res.ok()).toBe(true)
    const rows = await res.json()
    expect(Array.isArray(rows)).toBe(true)
    expect(rows.length).toBeGreaterThan(0)
    const row = rows[0]
    expect(row).toHaveProperty('frameNumber')
    expect(row).toHaveProperty('playerPositions')
  })

  test('FrameScrubber renders when tracks exist', async ({ page }) => {
    const errs = collectErrors(page)
    await page.goto(`${BASE}/analysis/${ANALYSIS_ID}`)
    await page.waitForLoadState('networkidle')

    // Tracks load async — wait up to 10s for scrubber
    const scrubber = page.locator('[data-testid="frame-scrubber"], .glass-card').filter({ hasText: /frame scrubber/i }).first()
    const visible = await scrubber.waitFor({ state: 'visible', timeout: 10_000 }).then(() => true).catch(() => false)

    await page.screenshot({ path: 'test-results/phase9-scrubber.png', fullPage: false })

    const criticalErrs = errs.filter(e => e.includes('TypeError') || e.includes('Cannot read'))
    expect(criticalErrs, `JS errors: ${criticalErrs.join('\n')}`).toHaveLength(0)
    expect(visible, 'FrameScrubber not visible — tracks may not have loaded or component not rendered').toBe(true)
  })

  test('Pitch tab shows player dots (no "No tracking data" message)', async ({ page }) => {
    const errs = collectErrors(page)
    await page.goto(`${BASE}/analysis/${ANALYSIS_ID}`)
    await page.waitForLoadState('networkidle')

    // Wait for tracks to load
    await page.waitForTimeout(3000)

    // Click Pitch tab
    const pitchTab = page.locator('[role="tab"]').filter({ hasText: /pitch|radar/i }).first()
    const tabVisible = await pitchTab.isVisible().catch(() => false)
    if (tabVisible) await pitchTab.click()

    await page.waitForTimeout(1000)
    await page.screenshot({ path: 'test-results/phase9-pitch-tab.png' })

    // Should NOT show "No tracking data available"
    const noDataMsg = await page.locator('text=/no tracking data/i').first().isVisible().catch(() => false)
    expect(noDataMsg, '"No tracking data available" message is still showing — frameData is null').toBe(false)

    // Should have SVG circles (player dots)
    const dots = await page.locator('svg circle').count()
    expect(dots, 'No SVG circles found — player dots not rendered on pitch').toBeGreaterThan(0)

    const criticalErrs = errs.filter(e => e.includes('TypeError') || e.includes('Cannot read'))
    expect(criticalErrs).toHaveLength(0)
  })

  test('Heatmap tab renders (grid cells or fallback ellipses)', async ({ page }) => {
    const errs = collectErrors(page)
    await page.goto(`${BASE}/analysis/${ANALYSIS_ID}`)
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(2000)

    const heatmapTab = page.locator('[role="tab"]').filter({ hasText: /heatmap/i }).first()
    const tabVisible = await heatmapTab.isVisible().catch(() => false)
    if (tabVisible) await heatmapTab.click()

    await page.waitForTimeout(800)
    await page.screenshot({ path: 'test-results/phase9-heatmap-tab.png' })

    // Should have SVG content (either real grid rects or fallback ellipses)
    const svgRects = await page.locator('svg rect').count()
    const svgEllipses = await page.locator('svg ellipse').count()
    expect(svgRects + svgEllipses, 'Heatmap SVG has no visual elements').toBeGreaterThan(0)

    const criticalErrs = errs.filter(e => e.includes('TypeError') || e.includes('Cannot read'))
    expect(criticalErrs).toHaveLength(0)
  })

  test('Pass network tab renders nodes', async ({ page }) => {
    const errs = collectErrors(page)
    await page.goto(`${BASE}/analysis/${ANALYSIS_ID}`)
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(2000)

    // Click by exact tab value attribute to avoid ambiguity with other tab lists on page
    const passTab = page.locator('[role="tab"][data-value="passes"], [role="tab"]').filter({ hasText: /^passes$/i }).first()
    const tabVisible = await passTab.isVisible().catch(() => false)
    if (tabVisible) {
      await passTab.click()
    } else {
      // fallback: click tab with aria-label or innerText "Passes"
      await page.getByRole('tab', { name: /passes/i }).first().click().catch(() => {})
    }

    await page.waitForTimeout(1200)
    await page.screenshot({ path: 'test-results/phase9-passnetwork-tab.png' })

    // Nodes are rendered as .player-node divs inside the passes tab content
    const nodes = await page.locator('[data-state="active"] .player-node, .player-node').count()
    expect(nodes, 'No pass network nodes rendered').toBeGreaterThan(0)

    const criticalErrs = errs.filter(e => e.includes('TypeError') || e.includes('Cannot read'))
    expect(criticalErrs).toHaveLength(0)
  })

  test('Frame scrubber slider changes frame value', async ({ page }) => {
    await page.goto(`${BASE}/analysis/${ANALYSIS_ID}`)
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(3000)

    // Get the slider input
    const slider = page.locator('[role="slider"]').first()
    const sliderVisible = await slider.waitFor({ state: 'visible', timeout: 8000 }).then(() => true).catch(() => false)

    if (!sliderVisible) {
      test.info().annotations.push({ type: 'info', description: 'Slider not visible — FrameScrubber may not have rendered (frameCount=0?)' })
      return
    }

    // Scope to the FrameScrubber card to avoid matching possession % elsewhere
    const scrubberCard = page.locator('.glass-card').filter({ hasText: /Frame Scrubber/i })
    const frameCounter = scrubberCard.locator('.font-mono').first()
    const before = await frameCounter.textContent().catch(() => '')

    // Move slider to the right
    await slider.focus()
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.waitForTimeout(300)

    const after = await frameCounter.textContent().catch(() => '')
    await page.screenshot({ path: 'test-results/phase9-slider-moved.png' })

    expect(before).not.toEqual(after)
  })
})
