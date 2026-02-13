import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        include: ['**/*.test.js', '**/*.test.mjs'],
        exclude: ['**/node_modules/**', '**/dist/**', '**/cypress/**', '**/.{idea,git,cache,output,temp}/**', '**/app/e2e/**'],
        environment: 'node',
    },
});
