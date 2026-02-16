import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MultiStreamManager, extractSpecFrame, SlotState } from './multi-stream-manager.js';
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
        });
    });

    describe('updatePeaks', () => {
        describe('近傍マッチング', () => {
            it('should match nearby peaks to existing slots', async () => {
                // Setup slots manually
                manager._slots.get('slot-0').freq = 800; // Sub
                manager._slots.get('slot-1').freq = 900; // Sub

                const peaks = [
                    { f: 805, snr: 10 },
                    { f: 905, snr: 10 },
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                expect(manager._slots.get('slot-0').freq).toBeCloseTo(800.5); // Sub smooths
                expect(manager._slots.get('slot-1').freq).toBeCloseTo(900.5); // Sub smooths
            });

            it('should skip peaks too close to main frequency', async () => {
                const peaks = [
                    { f: 705, snr: 10 }, // Too close to Main
                    { f: 800, snr: 10 },
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                // slot-0 should take 800
                expect(manager._slots.get('slot-0').freq).toBe(800);
            });
        });

        describe('新規ピークの割り当て', () => {
            it('should assign new peak to empty slot', async () => {
                const peaks = [
                    { f: 800, snr: 10 },
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                expect(manager._slots.get('slot-0').freq).toBe(800);
            });

            it('should replace worst slot when full and SNR is high enough', async () => {
                manager._slots.get('slot-0').freq = 800;
                manager._slots.get('slot-0').snr = 5;
                manager._slots.get('slot-0').lastTextUpdate = 0; // Very old (since START_TIME is 100000)

                // Fill other slots too to make it full
                manager._slots.get('slot-1').freq = 810; manager._slots.get('slot-1').snr = 100;
                manager._slots.get('slot-2').freq = 820; manager._slots.get('slot-2').snr = 100;
                manager._slots.get('slot-3').freq = 830; manager._slots.get('slot-3').snr = 100;

                mockNow.mockReturnValue(START_TIME + 25000); // 25s later

                const peaks = [
                    { f: 900, snr: 20 }, // Strong signal (20 > 5 * 2.0)
                ];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                // slot-0 should be replaced
                expect(manager._slots.get('slot-0').freq).toBe(900);
            });
        });

        describe('タイムアウト解放', () => {
            it('should timeout and release sub slots', async () => {
                manager._slots.get('slot-0').freq = 800;
                manager._slots.get('slot-0').lastSeen = START_TIME;
                manager._slots.get('slot-0').lastTextUpdate = 0;

                // Move forward 15s.
                // START_TIME = 100000.
                // now = 115000.
                // lastSeen = 100000. Diff = 15000 > 10000. Timeout OK.
                // lastText = 0. Diff = 115000 > 20000. Cooldown OK.
                mockNow.mockReturnValue(START_TIME + 15000);

                const peaks = [];
                const mainFreq = 700;

                await manager.updatePeaks(peaks, mainFreq);

                expect(manager._slots.get('slot-0').freq).toBe(0); // Released
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
            it('should calculate utilization per worker', async () => {
                // ワーカーごとのスロットIDは "worker-{workerIndex}-slot-{i}" の形式
                // テスト用のモックでは "slot-{i}" なので、ワーカーインデックスを抽出できない
                // そのため、このテストではワーカーインデックスを含むスロットIDを使用する
                
                // スロットIDをワーカーインデックスを含む形式に変更
                manager._slots.clear();
                manager._slots.set('worker-0-slot-0', new SlotState('worker-0-slot-0'));
                manager._slots.set('worker-0-slot-1', new SlotState('worker-0-slot-1'));
                manager._slots.set('worker-1-slot-0', new SlotState('worker-1-slot-0'));
                manager._slots.set('worker-1-slot-1', new SlotState('worker-1-slot-1'));

                // Worker 0: 2つのアクティブスロット、平均推論時間 40ms
                manager._slots.get('worker-0-slot-0').freq = 800;
                manager._slots.get('worker-0-slot-0').avgInferenceTime = 40;
                manager._slots.get('worker-0-slot-1').freq = 810;
                manager._slots.get('worker-0-slot-1').avgInferenceTime = 40;

                // Worker 1: 1つのアクティブスロット、平均推論時間 30ms
                manager._slots.get('worker-1-slot-0').freq = 900;
                manager._slots.get('worker-1-slot-0').avgInferenceTime = 30;

                const stats = await manager.performanceStats();

                // budget = chunkSize * hopMs = 12 * 10 = 120ms
                // Worker 0: (2 * 40) / 120 = 0.667 (66.7%)
                // Worker 1: (1 * 30) / 120 = 0.25 (25%)
                // 全体の使用率: max(0.667, 0.25) = 0.667
                expect(stats.utilization).toBeCloseTo(0.667, 2);
                expect(stats.activeSlots).toBe(3);
                
                // ワーカーごとの情報を検証
                expect(stats.workers).toHaveLength(2);
                expect(stats.workers[0].workerIndex).toBe(0);
                expect(stats.workers[0].activeSlots).toBe(2);
                expect(stats.workers[0].avgInferenceTime).toBe(40);
                expect(stats.workers[0].utilization).toBeCloseTo(0.667, 2);
                
                expect(stats.workers[1].workerIndex).toBe(1);
                expect(stats.workers[1].activeSlots).toBe(1);
                expect(stats.workers[1].avgInferenceTime).toBe(30);
                expect(stats.workers[1].utilization).toBeCloseTo(0.25, 2);
            });

            it('should exclude throttled slots from utilization calculation', async () => {
                manager._slots.clear();
                manager._slots.set('worker-0-slot-0', new SlotState('worker-0-slot-0'));
                manager._slots.set('worker-0-slot-1', new SlotState('worker-0-slot-1'));

                // 2つのスロットのうち、1つはスロットルされている
                manager._slots.get('worker-0-slot-0').freq = 800;
                manager._slots.get('worker-0-slot-0').avgInferenceTime = 40;
                manager._slots.get('worker-0-slot-0').throttled = false;

                manager._slots.get('worker-0-slot-1').freq = 810;
                manager._slots.get('worker-0-slot-1').avgInferenceTime = 40;
                manager._slots.get('worker-0-slot-1').throttled = true;  // スロットル中

                const stats = await manager.performanceStats();

                // スロットルされたスロットは除外されるため、1つのスロットのみ計算
                // (1 * 40) / 120 = 0.333 (33.3%)
                expect(stats.utilization).toBeCloseTo(0.333, 2);
                expect(stats.activeSlots).toBe(1);
                expect(stats.throttled).toBe(true);
                
                // ワーカーごとの情報を検証
                expect(stats.workers).toHaveLength(1);
                expect(stats.workers[0].activeSlots).toBe(1);
                expect(stats.workers[0].avgInferenceTime).toBe(40);
                expect(stats.workers[0].utilization).toBeCloseTo(0.333, 2);
            });

            it('should exclude unassigned slots (freq=0) from utilization calculation', async () => {
                manager._slots.clear();
                manager._slots.set('worker-0-slot-0', new SlotState('worker-0-slot-0'));
                manager._slots.set('worker-0-slot-1', new SlotState('worker-0-slot-1'));

                // 1つのスロットのみ割り当て
                manager._slots.get('worker-0-slot-0').freq = 800;
                manager._slots.get('worker-0-slot-0').avgInferenceTime = 40;

                // もう1つのスロットは未割り当て
                manager._slots.get('worker-0-slot-1').freq = 0;

                const stats = await manager.performanceStats();

                // 未割り当てのスロットは除外されるため、1つのスロットのみ計算
                // (1 * 40) / 120 = 0.333 (33.3%)
                expect(stats.utilization).toBeCloseTo(0.333, 2);
                expect(stats.activeSlots).toBe(1);
                
                // ワーカーごとの情報を検証
                expect(stats.workers).toHaveLength(1);
                expect(stats.workers[0].activeSlots).toBe(1);
                expect(stats.workers[0].avgInferenceTime).toBe(40);
                expect(stats.workers[0].utilization).toBeCloseTo(0.333, 2);
            });

            it('should return 0 utilization when no active slots', async () => {
                const stats = await manager.performanceStats();

                expect(stats.utilization).toBe(0);
                expect(stats.activeSlots).toBe(0);
                expect(stats.workers).toHaveLength(0);
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
