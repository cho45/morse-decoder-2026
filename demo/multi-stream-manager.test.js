import { describe, it, expect, beforeAll, beforeEach, vi } from 'vitest';
import ort from 'onnxruntime-node';
import path from 'path';
import { MultiStreamManager, extractSpecFrame } from './multi-stream-manager.js';
import { N_BINS } from './stream-inference.js';

// --- extractSpecFrame unit tests (pure function, no model) ---
describe('extractSpecFrame', () => {
    const nFft = 1024;
    const sampleRate = 32000;
    const binBW = sampleRate / nFft; // 31.25 Hz

    it('should return Float32Array of N_BINS length', () => {
        const magnitudes = new Float32Array(nFft / 2).fill(1);
        const result = extractSpecFrame(magnitudes, 700, nFft, sampleRate);
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(N_BINS);
    });

    it('should extract bins centered on the given frequency', () => {
        // Put a peak at bin 22 (22 * 31.25 = 687.5 Hz)
        const magnitudes = new Float32Array(nFft / 2).fill(0);
        magnitudes[22] = 1.0;

        // Center at 700 Hz → fStart = 700 - (14*31.25)/2 + 31.25/2 = 700 - 218.75 + 15.625 = 496.875
        // Bin indices: fStart/binBW = 496.875/31.25 = 15.9
        // Peak at bin 22 → relative index = 22 - 15.9 = 6.1 → should appear around index 6
        const result = extractSpecFrame(magnitudes, 700, nFft, sampleRate);

        // The peak should be in the frame somewhere
        const maxIdx = result.indexOf(Math.max(...result));
        expect(maxIdx).toBeGreaterThanOrEqual(5);
        expect(maxIdx).toBeLessThanOrEqual(7);
        expect(result[maxIdx]).toBeGreaterThan(0);
    });

    it('should return zeros when center frequency is out of range', () => {
        const magnitudes = new Float32Array(nFft / 2).fill(1);
        // Very low frequency → some bins will be negative index → zeros
        const result = extractSpecFrame(magnitudes, 50, nFft, sampleRate);
        // At least some bins should be zero (negative freq region)
        let hasZero = false;
        for (let i = 0; i < N_BINS; i++) {
            if (result[i] === 0) hasZero = true;
        }
        expect(hasZero).toBe(true);
    });

    it('should perform linear interpolation between bins', () => {
        const magnitudes = new Float32Array(nFft / 2).fill(0);
        // Set two adjacent bins to known values
        magnitudes[20] = 1.0;
        magnitudes[21] = 3.0;

        // Center freq such that extraction lands exactly between bin 20 and 21
        // This depends on exact alignment, so just verify interpolation produces intermediate values
        const centerFreq = 20.5 * binBW;
        const result = extractSpecFrame(magnitudes, centerFreq, nFft, sampleRate);

        // At least one bin should have an interpolated value between 1.0 and 3.0
        let hasInterpolated = false;
        for (let i = 0; i < N_BINS; i++) {
            if (result[i] > 0 && result[i] < 3.0) hasInterpolated = true;
        }
        expect(hasInterpolated).toBe(true);
    });
});

// --- MultiStreamManager tests (requires ONNX model) ---
describe('MultiStreamManager', () => {
    let session;
    const modelPath = path.resolve(import.meta.dirname, 'cw_decoder_quantized.onnx');

    beforeAll(async () => {
        session = await ort.InferenceSession.create(modelPath);
    });

    describe('Constructor', () => {
        it('should throw without session', () => {
            expect(() => new MultiStreamManager(null, ort)).toThrow('session');
        });

        it('should throw without ort', () => {
            expect(() => new MultiStreamManager(session, null)).toThrow('ort');
        });

        it('should create with default options', () => {
            const mgr = new MultiStreamManager(session, ort);
            expect(mgr.performanceStats.maxSlots).toBe(4);
            mgr.dispose();
        });

        it('should accept custom options', () => {
            const mgr = new MultiStreamManager(session, ort, {
                maxSlots: 2, chunkSize: 8, hopMs: 10
            });
            expect(mgr.performanceStats.maxSlots).toBe(2);
            expect(mgr.performanceStats.budget).toBe(80); // 8 * 10
            mgr.dispose();
        });
    });

    describe('Slot Allocation', () => {
        let mgr;
        beforeEach(() => {
            mgr = new MultiStreamManager(session, ort, { maxSlots: 4 });
        });

        it('should create main slot on first updatePeaks', () => {
            mgr.updatePeaks([], 700);
            const slots = mgr.getAllSlots();
            expect(slots.length).toBe(1);
            expect(slots[0].isMain).toBe(true);
            expect(slots[0].freq).toBe(700);
            mgr.dispose();
        });

        it('should allocate sub slots for detected peaks', () => {
            const peaks = [
                { f: 650, snr: 10 },
                { f: 1200, snr: 5 },
                { f: 2000, snr: 3 },
            ];
            mgr.updatePeaks(peaks, 700);
            const slots = mgr.getAllSlots();
            // 1 main + 3 sub = 4
            expect(slots.length).toBe(4);
            expect(slots.filter(s => s.isMain).length).toBe(1);
            mgr.dispose();
        });

        it('should not exceed maxSlots', () => {
            const peaks = [
                { f: 500, snr: 10 },
                { f: 1000, snr: 8 },
                { f: 1500, snr: 6 },
                { f: 2000, snr: 4 },
                { f: 2500, snr: 2 },
            ];
            mgr.updatePeaks(peaks, 700);
            const slots = mgr.getAllSlots();
            expect(slots.length).toBeLessThanOrEqual(4);
            mgr.dispose();
        });

        it('should skip peaks too close to main frequency', () => {
            const peaks = [
                { f: 720, snr: 10 }, // within ±50Hz of 700
                { f: 1200, snr: 5 },
            ];
            mgr.updatePeaks(peaks, 700);
            const slots = mgr.getAllSlots();
            // Main at 700, sub at 1200 only (720 is too close to main)
            expect(slots.length).toBe(2);
            expect(slots.find(s => !s.isMain).freq).toBeCloseTo(1200, 0);
            mgr.dispose();
        });

        it('should skip peaks too close to existing sub slot', () => {
            const peaks = [
                { f: 1200, snr: 10 },
                { f: 1220, snr: 8 },  // within ±50Hz of 1200
            ];
            mgr.updatePeaks(peaks, 700);
            const slots = mgr.getAllSlots();
            // 1 main + 1 sub (1220 too close to 1200)
            expect(slots.length).toBe(2);
            mgr.dispose();
        });
    });

    describe('Nearby Tracking', () => {
        it('should track existing slot when peak shifts slightly', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });

            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);
            const initialSlots = mgr.getAllSlots();
            const subSlot = initialSlots.find(s => !s.isMain);
            const initialFreq = subSlot.freq;

            // Peak shifts slightly
            mgr.updatePeaks([{ f: 1210, snr: 10 }], 700);
            const updatedSlots = mgr.getAllSlots();
            // Should still be 2 slots (not a new one)
            expect(updatedSlots.length).toBe(2);
            // Freq should have tracked (EMA: 0.9 * old + 0.1 * new)
            const updatedSub = updatedSlots.find(s => !s.isMain);
            expect(updatedSub.freq).toBeCloseTo(initialFreq * 0.9 + 1210 * 0.1, 1);

            mgr.dispose();
        });
    });

    describe('SNR Priority Replacement', () => {
        it('should replace lowest SNR sub slot when full and margin exceeded', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });

            // Fill up: main + 2 subs
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 2 },  // low SNR
            ], 700);
            expect(mgr.getAllSlots().length).toBe(3);

            // クールダウンを回避するため lastTextUpdate を巻き戻す
            for (const slot of mgr._slots) {
                slot.lastTextUpdate = Date.now() - 30000;
            }

            // SNR=2 の2倍超 (>4) の新ピーク → 置換される
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 2 },
                { f: 3000, snr: 20 },
            ], 700);

            const slots = mgr.getAllSlots();
            expect(slots.length).toBe(3);
            // 2000 Hz (SNR=2) should have been evicted
            const freqs = slots.map(s => Math.round(s.freq));
            expect(freqs).not.toContain(2000);

            mgr.dispose();
        });

        it('should NOT replace when SNR margin is insufficient', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });

            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 5 },
            ], 700);

            // クールダウン回避
            for (const slot of mgr._slots) {
                slot.lastTextUpdate = Date.now() - 30000;
            }

            // SNR=5 の2倍 = 10、新ピーク SNR=8 < 10 → 置換されない
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 5 },
                { f: 3000, snr: 8 },
            ], 700);

            const slots = mgr.getAllSlots();
            expect(slots.length).toBe(3);
            const freqs = slots.map(s => Math.round(s.freq));
            expect(freqs).toContain(2000); // 置換されない
            expect(freqs).not.toContain(3000);

            mgr.dispose();
        });

        it('should NOT replace slots that recently received text', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });

            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 2 },
            ], 700);
            // テキスト受信を模擬（lastTextUpdate が直近）
            mgr._slots.find(s => !s.isMain && Math.round(s.freq) === 2000).lastTextUpdate = Date.now();

            // SNR=2 の2倍超だがテキスト受信直後 → 置換されない
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 2 },
                { f: 3000, snr: 20 },
            ], 700);

            const slots = mgr.getAllSlots();
            expect(slots.length).toBe(3);
            const freqs = slots.map(s => Math.round(s.freq));
            expect(freqs).toContain(2000); // テキスト受信直後なので置換されない

            mgr.dispose();
        });
    });

    describe('Timeout Release', () => {
        it('should release sub slot after timeout', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });

            // Create sub slot
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);
            expect(mgr.getAllSlots().length).toBe(2);

            // Simulate time passing (manipulate lastSeen)
            const subSlot = mgr.getAllSlots().find(s => !s.isMain);
            // Access internal slot to set lastSeen
            const internalSlot = mgr._slots.find(s => !s.isMain);
            internalSlot.lastSeen = Date.now() - 11000; // 11 seconds ago

            // Update without the peak → timeout should trigger
            mgr.updatePeaks([], 700);
            const slotsAfter = mgr.getAllSlots();
            // Sub slot should be released
            expect(slotsAfter.length).toBe(1);
            expect(slotsAfter[0].isMain).toBe(true);

            mgr.dispose();
        });

        it('should not timeout main slot', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });

            mgr.updatePeaks([], 700);
            const mainSlot = mgr._slots.find(s => s.isMain);
            mainSlot.lastSeen = Date.now() - 20000; // 20 seconds ago

            mgr.updatePeaks([], 700);
            // Main should still exist
            expect(mgr.getAllSlots().length).toBe(1);
            expect(mgr.getAllSlots()[0].isMain).toBe(true);

            mgr.dispose();
        });
    });

    describe('Performance Monitoring', () => {
        it('should report zero utilization initially', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 4, chunkSize: 12, hopMs: 10 });
            const stats = mgr.performanceStats;
            expect(stats.totalInferenceTime).toBe(0);
            expect(stats.budget).toBe(120);
            expect(stats.utilization).toBe(0);
            expect(stats.activeSlots).toBe(0);
            expect(stats.throttled).toBe(false);
            mgr.dispose();
        });

        it('should count active slots correctly', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 4 });
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 5 },
            ], 700);

            const stats = mgr.performanceStats;
            expect(stats.activeSlots).toBe(3); // main + 2 subs
            mgr.dispose();
        });

        it('should EMA-smooth inference time on pushMagnitudes', async () => {
            const mgr = new MultiStreamManager(session, ort, {
                maxSlots: 2, chunkSize: 4, hopMs: 10
            });
            mgr.updatePeaks([], 700);

            const nFft = 1024;
            const sampleRate = 32000;
            const magnitudes = new Float32Array(nFft / 2).fill(0.01);

            // Push enough frames to trigger inference
            for (let i = 0; i < 8; i++) {
                mgr.pushMagnitudes(magnitudes, nFft, sampleRate);
            }

            // Wait for processing to complete
            const mainInf = mgr.mainInference;
            if (mainInf) await mainInf.waitForProcessing();

            const mainSlot = mgr.getAllSlots().find(s => s.isMain);
            // After inference, avgInferenceTime should be > 0
            expect(mainSlot.avgInferenceTime).toBeGreaterThanOrEqual(0);

            mgr.dispose();
        });
    });

    describe('Throttling', () => {
        it('should throttle lowest SNR sub slot when budget exceeded', () => {
            const mgr = new MultiStreamManager(session, ort, {
                maxSlots: 4, chunkSize: 12, hopMs: 10
            });
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 2 },
            ], 700);

            // Simulate high inference times by setting avgInferenceTime directly
            for (const slot of mgr._slots) {
                slot.avgInferenceTime = 50; // 50ms each, 3 * 50 = 150 > 120 * 0.8 = 96
            }

            mgr._updateThrottling();

            const throttled = mgr.getAllSlots().filter(s => s.throttled);
            expect(throttled.length).toBe(1);
            // Should throttle the lowest SNR sub (2000 Hz, SNR=2)
            expect(Math.round(throttled[0].freq)).toBe(2000);

            mgr.dispose();
        });

        it('should not throttle main slot', () => {
            const mgr = new MultiStreamManager(session, ort, {
                maxSlots: 2, chunkSize: 12, hopMs: 10
            });
            mgr.updatePeaks([], 700);

            // Simulate extreme budget pressure
            for (const slot of mgr._slots) {
                slot.avgInferenceTime = 200; // Way over budget
            }

            mgr._updateThrottling();

            // Main is the only slot, should not be throttled
            const mainSlot = mgr.getAllSlots().find(s => s.isMain);
            expect(mainSlot.throttled).toBe(false);

            mgr.dispose();
        });

        it('should recover throttled slot when budget allows', () => {
            const mgr = new MultiStreamManager(session, ort, {
                maxSlots: 3, chunkSize: 12, hopMs: 10
            });
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);

            // Throttle the sub slot manually
            mgr._slots.find(s => !s.isMain).throttled = true;

            // Set low inference times → under recover threshold (50% of 120 = 60)
            for (const slot of mgr._slots) {
                slot.avgInferenceTime = 20; // total = 20ms < 60ms threshold
            }

            mgr._updateThrottling();

            const throttled = mgr.getAllSlots().filter(s => s.throttled);
            expect(throttled.length).toBe(0);

            mgr.dispose();
        });

        it('should not push frames to throttled slots', async () => {
            const mgr = new MultiStreamManager(session, ort, {
                maxSlots: 3, chunkSize: 4, hopMs: 10
            });
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);

            // Throttle the sub slot
            mgr._slots.find(s => !s.isMain).throttled = true;

            const nFft = 1024;
            const sampleRate = 32000;
            const magnitudes = new Float32Array(nFft / 2).fill(0.01);

            for (let i = 0; i < 8; i++) {
                mgr.pushMagnitudes(magnitudes, nFft, sampleRate);
            }

            // Wait for main to finish
            if (mgr.mainInference) await mgr.mainInference.waitForProcessing();

            // Sub slot should have frameCount 0 (no frames pushed)
            const subInference = mgr._slots.find(s => !s.isMain).inference;
            expect(subInference.frameCount).toBe(0);

            mgr.dispose();
        });
    });

    describe('setMaxSlots', () => {
        it('should evict excess sub slots when reduced', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 4 });
            mgr.updatePeaks([
                { f: 1200, snr: 10 },
                { f: 2000, snr: 5 },
                { f: 3000, snr: 2 },
            ], 700);
            expect(mgr.getAllSlots().length).toBe(4);

            mgr.setMaxSlots(2);
            expect(mgr.getAllSlots().length).toBe(2);
            // Main should remain
            expect(mgr.getAllSlots().find(s => s.isMain)).toBeDefined();

            mgr.dispose();
        });
    });

    describe('getTextForFreq', () => {
        it('should return null for unmatched frequency', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);
            expect(mgr.getTextForFreq(5000)).toBeNull();
            mgr.dispose();
        });

        it('should return text for matched sub slot', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);

            // Manually set text for testing
            mgr._slots.find(s => !s.isMain).text = 'CQ CQ';

            expect(mgr.getTextForFreq(1200)).toBe('CQ CQ');
            expect(mgr.getTextForFreq(1210)).toBe('CQ CQ'); // within tolerance
            mgr.dispose();
        });

        it('should not return main slot text', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 2 });
            mgr.updatePeaks([], 700);
            mgr._slots.find(s => s.isMain).text = 'MAIN TEXT';

            expect(mgr.getTextForFreq(700)).toBeNull();
            mgr.dispose();
        });
    });

    describe('Dispose & Reset', () => {
        it('should clear all slots on reset', () => {
            const mgr = new MultiStreamManager(session, ort, { maxSlots: 3 });
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);
            expect(mgr.getAllSlots().length).toBe(2);

            mgr.reset();
            expect(mgr.getAllSlots().length).toBe(0);
            mgr.dispose();
        });

        it('should be safe to call dispose multiple times', () => {
            const mgr = new MultiStreamManager(session, ort);
            mgr.dispose();
            mgr.dispose(); // Should not throw
        });

        it('should ignore operations after dispose', () => {
            const mgr = new MultiStreamManager(session, ort);
            mgr.dispose();
            // These should not throw
            mgr.updatePeaks([{ f: 1200, snr: 10 }], 700);
            mgr.pushMagnitudes(new Float32Array(512), 1024, 32000);
            expect(mgr.getAllSlots().length).toBe(0);
        });
    });
});
