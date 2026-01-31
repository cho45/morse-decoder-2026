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
        it('should process remaining buffered frames', async () => {
            const inference = new StreamInference(session, ort, { chunkSize: 12 });
            const frame = new Float32Array(N_BINS).fill(0.1);

            // Feed fewer frames than chunk size
            for (let i = 0; i < 5; i++) {
                inference.pushFrame(frame);
            }

            expect(inference.bufferSize).toBe(5);

            // Flush should process and clear buffer
            await inference.flush();

            // Buffer should be processed (padded to multiple of 4 = 8)
            expect(inference.bufferSize).toBe(0);

            inference.dispose();
        });

        it('should handle empty buffer', async () => {
            const inference = new StreamInference(session, ort);

            // Should not throw
            await expect(inference.flush()).resolves.toBeUndefined();

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
            expect(cer).toBe(0);

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
});
