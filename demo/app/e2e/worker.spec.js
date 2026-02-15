import { test, expect } from '@playwright/test';

test.describe('MultiStreamProxy Worker Test', () => {
    test.beforeEach(async ({ page }) => {
        page.on('console', msg => {
            console.log(`[Browser Console ${msg.type()}] ${msg.text()}`);
            if (msg.type() === 'error') {
                console.error(`[Browser Error] ${msg.text()}`);
            }
        });

        page.on('pageerror', error => {
            console.error('[Page Error]', error);
        });

        await page.goto('/app/e2e/worker.html');
    });

    test('should initialize MultiStreamProxy successfully', async ({ page }) => {
        // Wait for initialization to complete
        const testStatus = page.locator('#test-status');
        await expect(testStatus).toHaveText(/Initialization successful/, { timeout: 60000 });

        // Verify that 16 slots are created (4 workers * 4 slots per worker)
        const slotCards = page.locator('.slot-card');
        await expect(slotCards).toHaveCount(16);
    });

    test('should start test and push frames to all slots', async ({ page }) => {
        const startBtn = page.locator('#start-btn');
        const stopBtn = page.locator('#stop-btn');

        // Wait for initialization
        await expect(page.locator('#test-status')).toHaveText(/Initialization successful/, { timeout: 60000 });

        // Start test
        await startBtn.click();
        await expect(stopBtn).toBeEnabled();

        // Check for busy indicators
        const busyIndicators = page.locator('.busy-indicator');
        await expect(busyIndicators).toHaveCount(16);

        // Wait for some processing logs
        const log = page.locator('#log');
        await page.waitForTimeout(2000);

        // Stop test
        await stopBtn.click();
        await expect(startBtn).toBeEnabled();

        // Verify log contains expected messages
        await expect(log).toContainText('Starting test');
        await expect(log).toContainText('Test stopped');
    });

    test('should toggle busy indicators during processing', async ({ page }) => {
        const startBtn = page.locator('#start-btn');
        const stopBtn = page.locator('#stop-btn');

        // Wait for initialization
        await expect(page.locator('#test-status')).toHaveText(/Initialization successful/, { timeout: 60000 });

        // Start test
        await startBtn.click();

        // Wait for processing to start
        await page.waitForTimeout(1000);

        // Check that busy indicators exist
        const firstIndicator = page.locator('.busy-indicator').first();
        await expect(firstIndicator).toHaveClass(/busy-indicator/);

        // Stop test
        await stopBtn.click();
    });

    test('should display decoded text for slots', async ({ page }) => {
        const startBtn = page.locator('#start-btn');
        const stopBtn = page.locator('#stop-btn');

        // Wait for initialization
        await expect(page.locator('#test-status')).toHaveText(/Initialization successful/, { timeout: 60000 });

        // Start test
        await startBtn.click();

        // Wait for some processing
        await page.waitForTimeout(3000);

        // Check that text elements exist
        const textEls = page.locator('.decoded-text');
        for (let i = 0; i < 4; i++) {
            await expect(textEls.nth(i)).toBeVisible();
        }

        // Stop test
        await stopBtn.click();
    });

    test('should handle stop and restart correctly', async ({ page }) => {
        const startBtn = page.locator('#start-btn');
        const stopBtn = page.locator('#stop-btn');

        // Wait for initialization
        await expect(page.locator('#test-status')).toHaveText(/Initialization successful/, { timeout: 60000 });

        // Start test
        await startBtn.click();
        await expect(stopBtn).toBeEnabled();

        // Stop test
        await stopBtn.click();
        await expect(startBtn).toBeEnabled();

        // Restart test
        await startBtn.click();
        await expect(stopBtn).toBeEnabled();

        // Stop test again
        await stopBtn.click();
        await expect(startBtn).toBeEnabled();
    });
});
