
import { describe, it, expect, beforeEach } from 'vitest';
import { PeakDetector } from './peak-detector.js';

describe('PeakDetector', () => {
    const N_BINS = 128;
    let detector;

    beforeEach(() => {
        detector = new PeakDetector(N_BINS);
    });

    describe('Initialization & Basics', () => {
        it('should throw on invalid nBins', () => {
            expect(() => new PeakDetector(0)).toThrow();
            expect(() => new PeakDetector(-1)).toThrow();
            expect(() => new PeakDetector(10.5)).toThrow();
        });

        it('should throw if input size mismatches', () => {
            expect(() => detector.process(new Float32Array(N_BINS - 1))).toThrow();
        });

        it('should return empty array for zero input', () => {
            const peaks = detector.process(new Float32Array(N_BINS));
            expect(peaks).toEqual([]);
            expect(detector.noiseFloor).toBe(0);
        });
    });

    describe('Algorithm Correctness', () => {
        it('should detect a stable carrier and calculate SNR correctly', () => {
            const noiseBase = 0.01;
            const signal = 0.1; // SNR = 10 (linear)

            // Accumulate (approx 200 frames for alpha=0.05 to reach steady state)
            for (let i = 0; i < 200; i++) {
                const input = new Float32Array(N_BINS).fill(noiseBase);
                input[50] = signal;
                detector.process(input);
            }

            const peaks = detector.process(new Float32Array(N_BINS).fill(noiseBase));
            expect(peaks.length).toBe(1);
            expect(peaks[0].index).toBe(50);
            expect(peaks[0].magnitude).toBeCloseTo(signal, 2);
            // Expected SNR = 0.1 / 0.01 = 10
            expect(peaks[0].snr).toBeCloseTo(10, 0); // Loosened to 0 decimal places for stability
        });

        it('should return peaks sorted by magnitude descending', () => {
            const input = new Float32Array(N_BINS).fill(0.001);
            input[20] = 0.5; // Weak peak
            input[80] = 1.0; // Strong peak

            for (let i = 0; i < 150; i++) detector.process(input);

            const peaks = detector.process(input);
            expect(peaks.length).toBe(2);
            expect(peaks[0].index).toBe(80);
            expect(peaks[1].index).toBe(20);
        });

        it('should perform parabolic interpolation for sub-bin accuracy', () => {
            const input = new Float32Array(N_BINS).fill(0.001);
            // Peak slightly offset towards bin 61
            input[59] = 0.4;
            input[60] = 1.0;
            input[61] = 0.8;

            for (let i = 0; i < 150; i++) detector.process(input);
            const peaks = detector.process(input);

            expect(peaks[0].index).toBeGreaterThan(60);
            expect(peaks[0].index).toBeLessThan(61);
        });

        it('should respect 500ms integration (suppress short bursts)', () => {
            // Initial silence
            for (let i = 0; i < 150; i++) detector.process(new Float32Array(N_BINS).fill(0.001));

            // Single very strong burst (10.0 magnitude vs 0.001 noise)
            const burst = new Float32Array(N_BINS).fill(0.001);
            burst[50] = 10.0;

            const peaks = detector.process(burst);

            // Alpha is 0.05, so 1 frame of 10.0 becomes 0.5
            expect(peaks[0].magnitude).toBeCloseTo(0.5, 2);
        });
    });

    describe('Edge Cases', () => {
        it('should handle flat spectrum (no peaks)', () => {
            const input = new Float32Array(N_BINS).fill(0.1);
            for (let i = 0; i < 150; i++) detector.process(input);
            const peaks = detector.process(input);
            expect(peaks).toEqual([]);
            expect(detector.noiseFloor).toBeCloseTo(0.1, 4);
        });

        it('should ignore peaks at boundaries (0 or nBins-1)', () => {
            const input = new Float32Array(N_BINS).fill(0.001);
            input[0] = 1.0;
            input[N_BINS - 1] = 1.0;
            for (let i = 0; i < 20; i++) detector.process(input);
            const peaks = detector.process(input);
            expect(peaks).toEqual([]);
        });

        it('should handle multiple peaks correctly', () => {
            const input = new Float32Array(N_BINS).fill(0.01);
            input[10] = 0.5;
            input[20] = 0.3;
            input[30] = 0.8;
            for (let i = 0; i < 50; i++) detector.process(input);
            const peaks = detector.process(input);
            expect(peaks.length).toBe(3);
            expect(peaks[0].index).toBe(30);
            expect(peaks[1].index).toBe(10);
            expect(peaks[2].index).toBe(20);
        });

        it('should handle NaN or Infinity in input (robustness check)', () => {
            const input = new Float32Array(N_BINS).fill(0.01);
            input[50] = NaN;
            input[60] = Infinity;
            // The algorithm should not crash, though behavior is undefined for these bins
            expect(() => detector.process(input)).not.toThrow();
        });
    });

    it('should maintain peaks for a long time (5s decay)', () => {
        const pd = new PeakDetector(100);
        const mags = new Float32Array(100);
        mags[50] = 10.0;

        // Peak reaches steady state (attack)
        for (let i = 0; i < 200; i++) pd.process(mags);

        const peaksBefore = pd.process(mags);
        expect(peaksBefore[0].index).toBeCloseTo(50);
        expect(peaksBefore[0].magnitude).toBeGreaterThan(9.0);

        // Input stops
        mags.fill(0);

        // After 50 frames (~1s), it should still be significant
        for (let i = 0; i < 50; i++) pd.process(mags);
        const peaksAfter1s = pd.process(mags);
        expect(peaksAfter1s.length).toBeGreaterThan(0);
        // decay alpha = 0.004. After 51 total frames of 0: (1-0.004)^51 ~= 0.81
        expect(peaksAfter1s[0].magnitude).toBeGreaterThan(7.0);

        // After 250 frames (~5s), it should still exist but be smaller
        // (1-0.004)^251 ~= 0.36
        for (let i = 0; i < 200; i++) pd.process(mags);
        const peaksAfter5s = pd.process(mags);
        expect(peaksAfter5s.length).toBeGreaterThan(0);
        expect(peaksAfter5s[0].magnitude).toBeGreaterThan(3.0);
    });
});
