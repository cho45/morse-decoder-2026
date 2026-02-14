import { test, expect } from '@playwright/test';

test.describe('ONNX Runtime Web Worker PoC', () => {
    test.beforeEach(async ({ page }) => {
        // Log console messages for debugging
        page.on('console', msg => {
            // Ignore some expected warnings like "bin width changed" or similar
            if (msg.type() === 'error') {
                console.error(`[Browser Error] ${msg.text()}`);
            } else {
                console.log(`[Browser Log] ${msg.text()}`);
            }
        });

        // Go to PoC page
        // Note: baseURL is set in playwright.config.js to http://localhost:3000
        await page.goto('/app/poc.html');
    });

    test('should initialize workers and run parallel inference', async ({ page }) => {
        // 1. Wait for "Start Parallel Inference" button to become enabled
        // This implies workers are initialized (Ready)
        const startBtn = page.locator('#start-btn');
        await expect(startBtn).toBeEnabled({ timeout: 30000 });

        // Verify initial state
        const workerStatuses = page.locator('.worker-status');
        await expect(workerStatuses).toHaveCount(4);

        // 2. Click Start
        await startBtn.click();

        // 3. Wait for all statuses to become "Done"
        for (let i = 0; i < 4; i++) {
            await expect(page.locator(`#status-${i}`)).toHaveText('Done', { timeout: 30000 });
        }

        // 4. Check log for specific success messages
        const log = page.locator('#log');
        await expect(log).toContainText('Worker 0 infer done');
        await expect(log).toContainText('Worker 3 infer done');
    });
});
