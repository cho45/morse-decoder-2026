import { describe, it, expect, beforeAll, beforeEach, vi } from 'vitest';
import ort from 'onnxruntime-node';
import path from 'path';
import { StreamInference, N_BINS, CHARS } from './stream-inference.js';
import { computeSpecFrames, HOP_LENGTH, calculateCER } from './inference.js';
import { MorseGenerator, SAMPLE_RATE } from './data_gen.js';

describe('StreamInference', () => {
    let session;
    const modelPath = path.resolve(__dirname, 'cw_decoder_quantized.onnx');

    beforeAll(async () => {
        session = await ort.InferenceSession.create(modelPath);
    });

    describe('Constructor', () => {
        it('should throw error without session', () => {
            expect(() => new StreamInference(null, ort)).toThrow('requires an ONNX session');
        });

        it('should throw error without ort', () => {
            expect(() => new StreamInference(session, null)).toThrow('requires ort module');
        });

        it('should throw error with invalid chunk size', () => {
            expect(() => new StreamInference(session, ort, { chunkSize: 10 })).toThrow('multiple of 4');
        });

        it('should create with default options', () => {
            const inference = new StreamInference(session, ort);
            expect(inference.frameCount).toBe(0);
            expect(inference.bufferSize).toBe(0);
            expect(inference.isProcessing).toBe(false);
            inference.dispose();
        });

        it('should create with custom options', () => {
            const inference = new StreamInference(session, ort, {
                chunkSize: 8,
                useWebGPU: true,
                historyLength: 100
            });
            expect(inference).toBeDefined();
            inference.dispose();
        });
    });

    describe('pushFrame', () => {
        it('should throw error with wrong frame size', () => {
            const inference = new StreamInference(session, ort);
            expect(() => inference.pushFrame(new Float32Array(10))).toThrow(`${N_BINS} bins`);
            inference.dispose();
        });

        it('should buffer frames until chunk size', () => {
            const inference = new StreamInference(session, ort, { chunkSize: 4 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            inference.pushFrame(frame);
            expect(inference.bufferSize).toBe(1);
            expect(inference.frameCount).toBe(1);

            inference.pushFrame(frame);
            expect(inference.bufferSize).toBe(2);

            inference.pushFrame(frame);
            expect(inference.bufferSize).toBe(3);

            // Fourth frame triggers inference
            inference.pushFrame(frame);
            // After inference, buffer should be empty
            expect(inference.frameCount).toBe(4);

            inference.dispose();
        });

        it('should increment frame count', () => {
            const inference = new StreamInference(session, ort, { chunkSize: 12 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            for (let i = 0; i < 10; i++) {
                inference.pushFrame(frame);
                expect(inference.frameCount).toBe(i + 1);
            }

            inference.dispose();
        });
    });

    describe('Events', () => {
        it('should emit frame event after chunk inference', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 4 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            const frameEvents = [];
            inference.addEventListener('frame', (e) => {
                frameEvents.push(e.detail);
            });

            for (let i = 0; i < 4; i++) {
                inference.pushFrame(frame);
            }

            // Wait for inference to complete
            await inference.waitForProcessing();

            expect(frameEvents.length).toBeGreaterThan(0);
            expect(frameEvents[0]).toHaveProperty('framePos');

            inference.dispose();
        });

        it('should emit result event when character is decoded', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 12 });

            const resultEvents = [];
            inference.addEventListener('result', (e) => {
                resultEvents.push(e.detail);
            });

            // Generate a real CW signal (use longer text to ensure proper decoding)
            const gen = new MorseGenerator(SAMPLE_RATE);
            const LOOKAHEAD_FRAMES = 30;
            const timing = gen.generateTiming('TEST', 25);
            const waveform = gen.generateWaveform(timing);

            // Add lookahead padding for model latency
            const lookaheadSamples = LOOKAHEAD_FRAMES * 160; // HOP_LENGTH
            const paddedWaveform = new Float32Array(waveform.length + lookaheadSamples);
            paddedWaveform.set(waveform);

            const specFrames = computeSpecFrames(paddedWaveform);

            // Feed frames
            const numFrames = specFrames.length / N_BINS;
            for (let i = 0; i < numFrames; i++) {
                const frame = specFrames.slice(i * N_BINS, (i + 1) * N_BINS);
                inference.pushFrame(frame);
            }

            // Wait for processing and flush remaining
            await inference.waitForProcessing();
            await inference.flush();

            // Should have decoded something
            const text = inference.getText();
            expect(text.length).toBeGreaterThan(0);

            inference.dispose();
        });
    });

    describe('History Management', () => {
        it('should maintain signal history within limit', async () => {
            const historyLength = 50;
            const inference = new StreamInference(session, ort, {
                chunkSize: 12,
                historyLength
            });
            const frame = new Float32Array(N_BINS).fill(0.1);

            // Feed many frames (multiple chunks)
            for (let i = 0; i < 96; i++) { // 96 = 12 * 8 chunks
                inference.pushFrame(frame);
            }

            // Wait for all inference to complete
            await inference.waitForProcessing();

            const sigHistory = inference.getSignalHistory();
            expect(sigHistory.length).toBeLessThanOrEqual(historyLength);

            inference.dispose();
        });

        it('should return copies of history arrays', () => {
            const inference = new StreamInference(session, ort);

            const sigHistory1 = inference.getSignalHistory();
            const sigHistory2 = inference.getSignalHistory();

            expect(sigHistory1).not.toBe(sigHistory2);

            inference.dispose();
        });
    });

    describe('Reset', () => {
        it('should clear all state on reset', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 4 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            // Feed frames and run inference
            for (let i = 0; i < 8; i++) {
                inference.pushFrame(frame);
            }
            await inference.waitForProcessing();

            expect(inference.frameCount).toBeGreaterThan(0);

            // Reset
            inference.reset();

            expect(inference.frameCount).toBe(0);
            expect(inference.bufferSize).toBe(0);
            expect(inference.getText()).toBe('');
            expect(inference.getEvents()).toHaveLength(0);
            expect(inference.getSignalHistory()).toHaveLength(0);
            expect(inference.getCTCHistory()).toHaveLength(0);
            expect(inference.getBoundaryHistory()).toHaveLength(0);

            inference.dispose();
        });
    });

    describe('Dispose', () => {
        it('should release all resources', () => {
            const inference = new StreamInference(session, ort);
            inference.dispose();

            // After dispose, these should return empty/default values
            expect(inference.getText()).toBe('');
            expect(inference.getEvents()).toHaveLength(0);
        });
    });

    describe('Multiple Instances', () => {
        it('should support multiple independent instances', async () => {
            const inference1 = new StreamInference(session, ort, { chunkSize: 4 });
            const inference2 = new StreamInference(session, ort, { chunkSize: 4 });

            const frame = new Float32Array(N_BINS).fill(0.1);

            // Feed frames to first instance
            for (let i = 0; i < 8; i++) {
                inference1.pushFrame(frame);
            }

            // Second instance should be independent
            expect(inference2.frameCount).toBe(0);
            expect(inference2.bufferSize).toBe(0);

            // Feed frames to second instance
            for (let i = 0; i < 4; i++) {
                inference2.pushFrame(frame);
            }

            // Wait for inference
            await inference1.waitForProcessing();
            await inference2.waitForProcessing();

            // Both should have independent state
            expect(inference1.frameCount).toBe(8);
            expect(inference2.frameCount).toBe(4);

            inference1.dispose();
            inference2.dispose();
        });
    });

    describe('Flush', () => {
        it('should process available multiples of 4 frames', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 12 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            // Feed 5 frames (less than chunkSize 12, but has one multiple of 4)
            for (let i = 0; i < 5; i++) {
                inference.pushFrame(frame);
            }

            expect(inference.bufferSize).toBe(5);

            // Flush should process 4 frames (floor(5/4)*4) and leave 1
            await inference.flush();

            expect(inference.bufferSize).toBe(1);
            // frameCount tracks input frames, so it should be 5
            expect(inference.frameCount).toBe(5);

            inference.dispose();
        });

        it('should handle empty buffer', async () => {
            const inference = new StreamInference(session, ort);
            // Should not throw
            await expect(inference.flush()).resolves.toBeUndefined();
            inference.dispose();
        });

        it('should recursively process accumulated frames', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 4 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            let lastFramePos = 0;
            inference.addEventListener('frame', (e) => {
                lastFramePos = e.detail.framePos;
            });

            // Push many frames quickly (20 frames = 5 chunks)
            for (let i = 0; i < 20; i++) {
                inference.pushFrame(frame);
            }

            // Wait for all processing to drain
            await inference.waitForProcessing();

            // Should have processed all 20 frames
            expect(inference.frameCount).toBe(20);
            expect(lastFramePos).toBe(20);
            expect(inference.bufferSize).toBe(0);

            inference.dispose();
        });
    });

    describe('End-to-End Decoding', () => {
        it('should decode "CQ" correctly', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 12 });

            // Generate CW signal
            const gen = new MorseGenerator(SAMPLE_RATE);
            const targetText = "CQ CQ ";
            const timing = gen.generateTiming(targetText, 25);
            const waveform = gen.generateWaveform(timing);

            // Add lookahead padding
            const LOOKAHEAD_FRAMES = 30;
            const lookaheadSamples = LOOKAHEAD_FRAMES * HOP_LENGTH;
            const paddedWaveform = new Float32Array(waveform.length + lookaheadSamples);
            paddedWaveform.set(waveform);

            const specFrames = computeSpecFrames(paddedWaveform);

            // Feed frames
            const numFrames = specFrames.length / N_BINS;
            for (let i = 0; i < numFrames; i++) {
                const frame = specFrames.slice(i * N_BINS, (i + 1) * N_BINS);
                inference.pushFrame(frame);
            }

            // Wait for processing and flush remaining
            await inference.waitForProcessing();
            await inference.flush();

            const decoded = inference.getText();
            const cer = calculateCER(targetText, decoded);
            // Allow trailing space difference (1 char / 6 = 0.166)
            expect(cer).toBeLessThanOrEqual(0.2);

            inference.dispose();
        });

        it('should decode "CQ DE K" with word spaces', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 12 });

            // Generate CW signal
            const gen = new MorseGenerator(SAMPLE_RATE);
            const targetText = "CQ DE K";
            const timing = gen.generateTiming(targetText, 25);
            const waveform = gen.generateWaveform(timing);

            // Add lookahead padding
            const LOOKAHEAD_FRAMES = 30;
            const lookaheadSamples = LOOKAHEAD_FRAMES * HOP_LENGTH;
            const paddedWaveform = new Float32Array(waveform.length + lookaheadSamples);
            paddedWaveform.set(waveform);

            const specFrames = computeSpecFrames(paddedWaveform);

            // Feed frames
            const numFrames = specFrames.length / N_BINS;
            for (let i = 0; i < numFrames; i++) {
                const frame = specFrames.slice(i * N_BINS, (i + 1) * N_BINS);
                inference.pushFrame(frame);
            }

            // Wait for processing and flush remaining
            await inference.waitForProcessing();
            await inference.flush();

            const decoded = inference.getText();
            const cer = calculateCER(targetText, decoded);
            // Allow small CER for streaming (spacing might differ slightly)
            expect(cer).toBeLessThanOrEqual(0.3);

            // Verify events include positions
            const events = inference.getEvents();
            expect(events.length).toBeGreaterThan(0);
            events.forEach(ev => {
                expect(ev).toHaveProperty('char');
                expect(ev).toHaveProperty('pos');
            });

            inference.dispose();
        });
    });
    describe('Regression: Race Condition on Reset', () => {
        // Mocks specifically for this test to control timing
        class MockTensor {
            constructor(type, data, dims) {
                this.type = type;
                this.data = data;
                this.dims = dims;
            }
            dispose() { }
        }

        class MockSession {
            constructor() {
                this.runResolve = null;
                this.runPromise = null;
            }

            async run(inputs) {
                return new Promise(resolve => {
                    this.runResolve = resolve;
                });
            }

            // Helper to manually complete the inference
            completeRun(nextStates) {
                if (this.runResolve) {
                    const outputs = {
                        logits: new MockTensor('float32', new Float32Array(10 * 64), [1, 10, 64]),
                        signal_logits: new MockTensor('float32', new Float32Array(10 * 4), [1, 10, 4]),
                        boundary_logits: new MockTensor('float32', new Float32Array(10), [1, 10]),
                    };
                    // Map nextStates to new_* keys
                    for (const key of Object.keys(nextStates)) {
                        outputs[`new_${key}`] = nextStates[key];
                    }
                    this.runResolve(outputs);
                    this.runResolve = null;
                }
            }
        }

        const mockOrt = {
            Tensor: MockTensor
        };

        it('should prevent stale inference results from overwriting reset state', async () => {
            const session = new MockSession();
            const streamInference = new StreamInference(session, mockOrt, { chunkSize: 4, historyLength: 10 });

            // 1. Initial State
            const initialStates = streamInference._states;

            // 2. Push frames to trigger inference
            const frame = new Float32Array(N_BINS).fill(0.1);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);

            // Wait a macro task to ensure _runChunk execution started and hit await session.run()
            await new Promise(r => setTimeout(r, 0));

            expect(streamInference.isProcessing).toBe(true);

            // 3. Trigger Reset (Simulate 'redecode' during drag)
            streamInference.reset();

            // Verify reset happened
            const resetStates = streamInference._states;
            expect(initialStates).not.toBe(resetStates);

            // 4. Complete the delayed inference with "TAINTED" states
            const taintedStates = {
                pcen_state: new MockTensor('float32', new Float32Array([999]), [1, 1, 1]), // Marker
                sub_cache: new MockTensor('float32', new Float32Array([999]), [1, 1, 1]),
            };
            // Add other required keys
            for (let i = 0; i < 6; i++) {
                taintedStates[`attn_k_${i}`] = new MockTensor('float32', new Float32Array(0), []);
                taintedStates[`attn_v_${i}`] = new MockTensor('float32', new Float32Array(0), []);
                taintedStates[`offset_${i}`] = new MockTensor('int64', new BigInt64Array([0n]), []);
                taintedStates[`conv_cache_${i}`] = new MockTensor('float32', new Float32Array(0), []);
            }

            session.completeRun(taintedStates);

            // Wait for the async chain to finish
            await new Promise(r => setTimeout(r, 0));

            // 5. Verification
            const currentStates = streamInference._states;

            // If the bug exists, currentStates.pcen_state will be our tainted [999] tensor
            if (currentStates.pcen_state.data[0] === 999) {
                expect.fail("Race condition reproduced! _states was overwritten by old inference result.");
            } else {
                expect(currentStates.pcen_state.data[0]).not.toBe(999);
            }
        });
    });

    describe('Regression: Detached State Reuse on Failure', () => {
        // バッファのdetachと推論失敗をシミュレートするMock
        class MockTensor {
            constructor(type, data, dims) {
                this.type = type;
                this.data = data;
                this.dims = dims;
            }
            dispose() { }
        }

        class MockSession {
            async run(inputs) {
                // Detachmentシミュレーション: 入力のバッファがdetachされる (byteLength 0になる)
                Object.values(inputs).forEach(t => {
                    if (t.data && t.data.buffer) {
                        try {
                            // Node/V8ではWorkerへの転送やモックでdetachを再現できる。
                            // ここではモックデータを操作して "detached" (byteLength 0) 状態にする。
                            // MockTensorがデータを保持している前提。
                            // 具体的なエラー "Tensor's size(14) does not match data length(0)" を再現するには、
                            // 既存のテンソルのデータ長が0になる必要がある。
                            if (t.data.length > 0) {
                                // 空配列に置き換えて "Detach" 状態を模倣する。
                                // テンソルオブジェクト自体は生存している。
                                // 実際のWASMでは内部バッファがdetachされるが、
                                // inference.jsのコードは t.data.byteLength をチェックするため、
                                // ここではデータを空配列に置換することで再現する。
                                t.data = new Float32Array(0);
                            }
                        } catch (e) { }
                    }
                });

                // session.run 内でのエラー（ネットワークエラーや推論失敗など）をシミュレート
                throw new Error("Simulated Inference Failure");
            }
        }

        const mockOrt = {
            Tensor: MockTensor
        };

        it('should recover from failed inference where states were detached', async () => {
            const session = new MockSession();
            const streamInference = new StreamInference(session, mockOrt, { chunkSize: 4 });

            // 1. 初期状態 (正常)
            // pcen_state は [1, 1, 14] でデータが存在する
            const initialStates = streamInference._states;
            expect(initialStates.pcen_state.data.byteLength).toBeGreaterThan(0);

            // 2. 推論を実行し、失敗と入力のDETACHを引き起こす
            const frame = new Float32Array(N_BINS).fill(0.1);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);

            // 処理待ち（内部で例外が発生し、catch/logされるはず）
            await streamInference.waitForProcessing();

            // 3. この時点で、修正によりエラーが捕捉され reset() が呼ばれているはずである。
            // したがって _states は新規作成（初期化）されており、有効（byteLength > 0）であるべき。
            expect(streamInference._states.pcen_state.data.byteLength).toBeGreaterThan(0); // リカバリ済み
            expect(streamInference._states).not.toBe(initialStates); // リセットされた

            // 4. 再度推論を実行
            // 状態がリセットされているため、これは成功するはずである
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);
            streamInference.pushFrame(frame);

            await expect(streamInference.waitForProcessing()).resolves.not.toThrow();
        });
    });

    describe('Regression: Concurrent Redecode (Race Condition)', () => {
        class MockTensor {
            constructor(type, data, dims) {
                this.type = type;
                this.data = data;
                this.dims = dims;
            }
            dispose() { }
        }

        class SlowSession {
            constructor() {
                this.runningCount = 0;
            }
            async run(inputs) {
                this.runningCount++;
                if (this.runningCount > 1) {
                    throw new Error("Concurrency Error: session.run called while another run is pending!");
                }
                // 推論の遅延をシミュレート (10ms)
                await new Promise(r => setTimeout(r, 10));
                this.runningCount--;

                // ダミー出力を返す
                const outputs = {
                    logits: new MockTensor('float32', new Float32Array(10 * 64), [1, 10, 64]),
                    signal_logits: new MockTensor('float32', new Float32Array(10 * 4), [1, 10, 4]),
                    boundary_logits: new MockTensor('float32', new Float32Array(10), [1, 10]),
                };
                // nextStates を入力からコピーして返す (ダミー)
                ['pcen_state', 'sub_cache'].forEach(k => outputs[`new_${k}`] = inputs[k]);
                for (let i = 0; i < 6; i++) {
                    ['attn_k', 'attn_v', 'offset', 'conv_cache'].forEach(k => outputs[`new_${k}_${i}`] = inputs[`${k}_${i}`]);
                }
                return outputs;
            }
        }
        const mockOrt = { Tensor: MockTensor };

        it('should NOT run concurrent inference loops when reset is called mid-processing', async () => {
            const consoleSpy = vi.spyOn(console, 'error');
            const session = new SlowSession();
            const inference = new StreamInference(session, mockOrt, { chunkSize: 4 });

            // 1. 推論1を開始
            const frame = new Float32Array(N_BINS).fill(0.1);
            inference.pushFrame(frame);
            inference.pushFrame(frame);
            inference.pushFrame(frame);
            inference.pushFrame(frame); // ここで _runChunk -> session.run (10ms待機) がトリガーされる

            expect(inference.isProcessing).toBe(true);

            // 2. 直ちにリセットし、推論2を開始 (redecodeのシミュレーション)
            // バグがある場合、reset() が isProcessing=false に設定してしまうため、
            // pushFrame が2回目の _runChunk を起動してしまい、並行実行が発生する。
            inference.reset();

            // 2回目のストリーム用にフレームをプッシュ
            inference.pushFrame(frame);
            inference.pushFrame(frame);
            inference.pushFrame(frame);
            inference.pushFrame(frame); // 安全ならばキューイングされるか、前の完了を待つべき

            // 処理が落ち着くまで待機
            await new Promise(r => setTimeout(r, 50));

            // コンソールエラーに "Concurrency Error" が含まれていないことを確認
            const errors = consoleSpy.mock.calls.map(args => args.join(' '));
            const concurrencyErrors = errors.filter(e => e.includes('Concurrency Error'));
            expect(concurrencyErrors).toHaveLength(0);

            inference.dispose();
            consoleSpy.mockRestore();
        });
    });
});
