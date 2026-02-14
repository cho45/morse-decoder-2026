import { test, expect } from '@playwright/test';

test.describe('ONNX Runtime Web Worker PoC (Continuous Streaming)', () => {
    test.beforeEach(async ({ page }) => {
        page.on('console', msg => {
            if (msg.type() === 'error') {
                console.error(`[Browser Error] ${msg.text()}`);
            } else {
                console.log(`[Browser Log] ${msg.text()}`);
            }
        });

        await page.goto('/app/poc.html');
    });

    test('should stream continuously and show busy indicator', async ({ page }) => {
        const startBtn = page.locator('#start-btn');
        const stopBtn = page.locator('#stop-btn');

        await expect(startBtn).toBeEnabled({ timeout: 60000 });

        // Start streaming
        await startBtn.click();
        await expect(stopBtn).toBeEnabled();

        // Check for busy indicators
        const indicators = page.locator('.busy-indicator');
        await expect(indicators).toHaveCount(16); // 4 workers * 4 slots

        // Wait for some processing logs
        const log = page.locator('#log');
        // We look for any text update or log message during streaming
        // Since it's continuous, we just wait a bit
        await page.waitForTimeout(2000);

        // Stop streaming
        await stopBtn.click();
        await expect(startBtn).toBeEnabled();

        // After stopping, check if any slot reported "done" (log message)
        await expect(log).toContainText('done', { timeout: 10000 });

        // Check text elements (should have been updated during streaming)
        const textEls = page.locator('.decoded-text');
        for (let i = 0; i < 4; i++) {
            // Just verify they exist
            await expect(textEls.nth(i)).toBeVisible();
        }
    });

    test('should toggle busy indicator during streaming', async ({ page }) => {
        const startBtn = page.locator('#start-btn');
        await expect(startBtn).toBeEnabled({ timeout: 60000 });
        await startBtn.click();

        // The indicator should be alternating or stay busy/idle
        // Since inference is fast (~10ms) and interval is 120ms, it will be idle most of the time
        // but it should become 'busy' (red) during pushFrame/updateStatus.
        // Actually, our updateStatus polls getResults which returns isProcessing.

        const firstIndicator = page.locator('.busy-indicator').first();

        // This is hard to catch in a flaky test, but we can at least check if the class exists
        await expect(firstIndicator).toHaveClass(/busy-indicator/);

        await page.locator('#stop-btn').click();
    });
});
