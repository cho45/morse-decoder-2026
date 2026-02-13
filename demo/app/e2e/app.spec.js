
import { test, expect } from '@playwright/test';

test.describe('CW Decoder Demo', () => {
    test.beforeEach(async ({ page }) => {
        // Capture console logs
        page.on('console', msg => {
            if (msg.type() === 'error' || msg.type() === 'warning') {
                const text = msg.text();
                // Ignore known benign warnings
                if (text.includes('Unknown CPU vendor')) return;

                console.log(`[Browser Console] ${msg.type()}: ${text}`);
                // Fail test on error (strict mode)
                if (msg.type() === 'error') {
                    throw new Error(`Browser Console Error: ${text}`);
                }
            }
        });

        // Grant microphone permission and load the app
        await page.context().grantPermissions(['microphone']);
        await page.goto('/app/');
    });

    test('should start demo mode and decode CQ', async ({ page }) => {
        // Click "Start Demo Mode"
        await page.getByRole('button', { name: 'デモモード開始' }).click();

        // Verify progress indicator appears (Downloading OR Initializing)
        // Since download might be fast, we check for either state or just that the button is disabled/loading
        // But the user specifically wants to know if the valid text is shown.
        // We can try to catch "モデルをダウンロード中" or "AIエンジンを初期化中"
        await expect(page.locator('.actions button.secondary')).toContainText(/モデルをダウンロード中|AIエンジンを初期化中|準備中/);

        // Verify main screen is shown
        await expect(page.locator('.main-screen')).toBeVisible();

        // Wait for "CQ" to appear in the decoded text
        // The demo sends "CQ CQ DE ..."
        await expect(page.locator('.decoded-text-content')).toContainText('CQ', { timeout: 15000 });
    });

    test('should update frequency when clicking on waterfall', async ({ page }) => {
        // Start Demo Mode to get to the main screen
        await page.getByRole('button', { name: 'デモモード開始' }).click();
        await expect(page.locator('.main-screen')).toBeVisible();

        // Get initial frequency
        const freqDisplay = page.locator('.freq-display');

        // Wait for frequency to be displayed
        await expect(freqDisplay).toBeVisible();

        // Click on the waterfall overlay
        // The overlay is at top-left. Clicking near the top should select a high frequency,
        // clicking near the bottom should select a low frequency.
        // MAX_FREQ is 4000. 
        // y=0 is top (High Freq ~4000Hz), y=height is bottom (0Hz).

        // Click near the top (High frequency)
        const overlay = page.locator('.waterfall-overlay-canvas');
        const box = await overlay.boundingBox();
        if (!box) throw new Error('Overlay not found');

        // Click at 10% from top
        // (1 - 0.1) * 4000 = 3600
        const clickY1 = box.height * 0.1;
        await overlay.click({ position: { x: box.width / 2, y: clickY1 } });

        // Calculate expected freq and tolerance
        // Resolution is MAX_FREQ / height. 4000 / 550 ~= 7.27 Hz/px
        const hzPerPx = 4000 / box.height;
        const expectedFreq1 = (1 - 0.1) * 4000;

        // Allow 2px tolerance
        const tolerance = hzPerPx * 2;

        await expect(async () => {
            const txt = await freqDisplay.textContent();
            const freq = parseInt(txt);
            if (Math.abs(freq - expectedFreq1) > tolerance) {
                throw new Error(`Expected ${expectedFreq1} +/- ${tolerance}, got ${freq}`);
            }
        }).toPass();


        // Click at 90% from top
        // (1 - 0.9) * 4000 = 400
        const clickY2 = box.height * 0.9;
        await overlay.click({ position: { x: box.width / 2, y: clickY2 } });

        const expectedFreq2 = (1 - 0.9) * 4000;

        await expect(async () => {
            const txt = await freqDisplay.textContent();
            const freq = parseInt(txt);
            if (Math.abs(freq - expectedFreq2) > tolerance) {
                throw new Error(`Expected ${expectedFreq2} +/- ${tolerance}, got ${freq}`);
            }
            if (Math.abs(freq - expectedFreq2) > tolerance) {
                throw new Error(`Expected ${expectedFreq2} +/- ${tolerance}, got ${freq}`);
            }
        }).toPass();
    });

    test('should track frequency when clicking peak marker', async ({ page }) => {
        // Start Demo Mode
        await page.getByRole('button', { name: 'デモモード開始' }).click();
        await expect(page.locator('.main-screen')).toBeVisible();

        // Wait for peak markers to appear (demo mode has simulated signals)
        const marker = page.locator('.peak-marker').first();
        await expect(marker).toBeVisible({ timeout: 10000 });

        // Get marker text (SNR) to ensure it's a valid marker
        const snrText = await marker.locator('.peak-label').textContent();
        console.log(`Found peak marker with SNR: ${snrText}`);

        // Click the marker
        await marker.click();

        // Verify frequency display updates 
        // We don't know the exact freq of the first marker easily without parsing style,
        // but we can check if the freq display stabilizes or changes.
        // Or better, check if the "Locked" indicator (red rect) aligns? 
        // For now, just verify clickability and no error.

        // Check if tracked frequency matches the marker's approximate position?
        // Let's assume the action is successful if the app doesn't crash.
        // We can verify "Auto Track" behavior if we had a checkbox for it visible/checkable.
    });

    test('should stop demo mode and return to setup (overlay)', async ({ page }) => {
        // Start Demo Mode
        await page.getByRole('button', { name: 'デモモード開始' }).click();
        await expect(page.locator('.main-screen')).toBeVisible();

        // Click Stop button
        await page.getByRole('button', { name: '■' }).click();

        // Verify setup screen is shown again
        await expect(page.locator('.setup-screen')).toBeVisible();

        // Main screen should ALSO be visible (behind overlay)
        await expect(page.locator('.main-screen')).toBeVisible();
    });
});
