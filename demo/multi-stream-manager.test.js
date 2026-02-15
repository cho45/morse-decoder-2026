import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MultiStreamManager, extractSpecFrame } from './multi-stream-manager.js';
import { MockMultiStreamProxy } from './mocks/MockMultiStreamProxy.js';

describe('MultiStreamManager', () => {
    let manager;
    let mockProxy;
    let mockNow;
    const START_TIME = 100000;

    beforeEach(async () => {
        mockProxy = new MockMultiStreamProxy();
        mockNow = vi.fn();
        mockNow.mockReturnValue(START_TIME);

        manager = new MultiStreamManager(
            new ArrayBuffer(100),
            {
                numWorkers: 2,
                maxSlots: 4,
                multiStreamProxy: mockProxy,
                now: mockNow,
            }
        );

        for (let i = 0; i < 4; i++) {
            await mockProxy.createSlot(`slot-${i}`, 0, {});
        }

        await manager.init();
    });

    describe('Constructor & Init', () => {
        it('should throw error without modelBuffer', () => {
            expect(() => new MultiStreamManager(null)).toThrow('requires modelBuffer');
        });

        it('should initialize slots from proxy', () => {
            expect(manager._slots.size).toBe(4);
            expect(manager.getMainSlotId()).toBe('slot-0'); // First one becomes main
        });
    });

    describe('updatePeaks', () => {
        describe('メインスロットの確保', () => {
            it('should update main slot frequency', async () => {
                const peaks = [{ f: 710, snr: 10 }];
                const mainFreq = 710;

                await manager.updatePeaks(peaks, mainFreq);

                const mainSlot = manager._slots.get('slot-0');
                expect(mainSlot.freq).toBe(710);
            });
        });

        describe('近傍マッチング', () => {
            it('should match nearby peaks to existing slots', async () => {
                // Setup slots manually
                manager._slots.get('slot-0').freq = 700; // Main
                manager._slots.get('slot-1').freq = 800; // Sub

                const peaks = [
                    { f: 705, snr: 10 },
                    { f: 805, snr: 10 },
                ];
                const mainFreq = 705;

                await manager.updatePeaks(peaks, mainFreq);

                expect(manager._slots.get('slot-0').freq).toBe(705); // Main tracks exactly
                expect(manager._slots.get('slot-1').freq).toBeCloseTo(800.5); // Sub smooths
            });

            it('should skip peaks too close to main frequency', async () => {
                manager._slots.get('slot-0').freq = 700;
                const peaks = [
                    { f: 705, snr: 10 }, // Too close to Main
                    { f: 800, snr: 10 },
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                // slot-1 should take 800
                expect(manager._slots.get('slot-1').freq).toBe(800);
            });
        });

        describe('新規ピークの割り当て', () => {
            it('should assign new peak to empty slot', async () => {
                manager._slots.get('slot-0').freq = 700;
                const peaks = [
                    { f: 800, snr: 10 },
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                expect(manager._slots.get('slot-1').freq).toBe(800);
            });

            it('should replace worst slot when full and SNR is high enough', async () => {
                manager._slots.get('slot-0').freq = 700;
                manager._slots.get('slot-1').freq = 800;
                manager._slots.get('slot-1').snr = 5;
                manager._slots.get('slot-1').lastTextUpdate = 0; // Very old (since START_TIME is 100000)

                // Fill other slots too to make it full
                manager._slots.get('slot-2').freq = 810; manager._slots.get('slot-2').snr = 100;
                manager._slots.get('slot-3').freq = 820; manager._slots.get('slot-3').snr = 100;

                mockNow.mockReturnValue(START_TIME + 25000); // 25s later

                const peaks = [
                    { f: 900, snr: 20 }, // Strong signal (20 > 5 * 2.0)
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                // slot-1 should be replaced
                expect(manager._slots.get('slot-1').freq).toBe(900);
            });
        });

        describe('タイムアウト解放', () => {
            it('should timeout and release sub slots', async () => {
                manager._slots.get('slot-0').freq = 700;
                manager._slots.get('slot-1').freq = 800;
                manager._slots.get('slot-1').lastSeen = START_TIME;
                manager._slots.get('slot-1').lastTextUpdate = 0;

                // Move forward 15s. 
                // START_TIME = 100000. 
                // now = 115000.
                // lastSeen = 100000. Diff = 15000 > 10000. Timeout OK.
                // lastText = 0. Diff = 115000 > 20000. Cooldown OK.
                mockNow.mockReturnValue(START_TIME + 15000);

                const peaks = [];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                expect(manager._slots.get('slot-1').freq).toBe(0); // Released
                expect(manager._slots.get('slot-0').freq).toBe(700); // Main preserved
            });
        });

        describe('AsyncLock (Race Condition)', () => {
            it('should skip concurrent updatePeaks calls', async () => {
                let callCount = 0;
                // Original method replacement for counting
                const originalUpdate = manager._updatePeaksInternal;
                manager._updatePeaksInternal = async (peaks, mainFreq) => {
                    callCount++;
                    await new Promise(r => setTimeout(r, 10)); // Delay
                };

                const p1 = manager.updatePeaks([], 700);
                const p2 = manager.updatePeaks([], 700);
                const p3 = manager.updatePeaks([], 700);

                await Promise.all([p1, p2, p3]);

                // 3回呼んでも、p1実行中にp2, p3が来るのでロックされスキップされるはず
                // ただしp2がスキップされた後、p3が来るタイミングによっては...
                // 同時実行 (Promise.all) ならほぼ確実にかぶる
                // 1回しか実行されないことを期待
                expect(callCount).toBe(1);
            });
        });

        describe('パフォーマンス統計', () => {
            it('should return correct statistics', async () => {
                manager._slots.get('slot-0').freq = 700;
                manager._slots.get('slot-0').avgInferenceTime = 50;
                manager._slots.get('slot-1').freq = 800;
                manager._slots.get('slot-1').avgInferenceTime = 70;

                const stats = await manager.performanceStats();

                expect(stats.totalInferenceTime).toBe(120);
                expect(stats.activeSlots).toBe(2);
            });
        });
    });

    describe('extractSpecFrame', () => {
        it('should extract correct size spectrogram frame', () => {
            const magnitudes = new Float32Array(512).fill(1.0);
            const centerFreq = 700;
            const nFft = 512;
            const sampleRate = 16000;

            const specFrame = extractSpecFrame(magnitudes, centerFreq, nFft, sampleRate);

            expect(specFrame.length).toBe(14); // N_BINS
        });
    });
});
