import { describe, it, expect, beforeAll } from 'vitest';
import ort from 'onnxruntime-node';
import path from 'path';
import { initStates, runFullInference, decodeFull, N_BINS, CHARS } from './inference.js';

describe('Inference Logic', () => {
    let session;
    const modelPath = path.resolve(__dirname, 'cw_decoder_quantized.onnx');

    beforeAll(async () => {
        session = await ort.InferenceSession.create(modelPath);
    });

    it('should initialize states correctly', () => {
        const states = initStates(ort);
        expect(states).toHaveProperty('sub_cache');
        expect(states.sub_cache.dims).toEqual([1, 1, 2, N_BINS]);
        expect(states).toHaveProperty('attn_k_0');
        expect(states.attn_k_0.dims).toEqual([1, 4, 0, 64]);
    });

    it('should run full inference on dummy frames', async () => {
        const tLen = 8; // Multiple of 4
        const specFrames = new Float32Array(tLen * N_BINS).fill(0.1);
        
        const { logits, signal_logits, numClasses } = await runFullInference(session, specFrames, ort);
        
        expect(logits).toBeInstanceOf(Float32Array);
        expect(signal_logits).toBeInstanceOf(Float32Array);
        expect(numClasses).toBe(CHARS.length + 1);
        
        // Subsampling rate is 2, so 8 frames -> 4 output frames
        expect(logits.length / numClasses).toBe(4);
    });

    it('should decode full logits correctly', () => {
        const numClasses = CHARS.length + 1;
        const T = 3;
        const allLogits = new Float32Array(T * numClasses).fill(-10);
        const allSignalLogits = new Float32Array(T * 4).fill(-10);

        // Frame 0: Character 'A'
        const aIndex = CHARS.indexOf('A') + 1;
        allLogits[aIndex] = 10;
        
        // Frame 1: Space (class 3)
        allSignalLogits[1 * 4 + 3] = 10;

        // Frame 2: Character 'B'
        const bIndex = CHARS.indexOf('B') + 1;
        allLogits[2 * numClasses + bIndex] = 10;

        const decoded = decodeFull(allLogits, allSignalLogits, numClasses);
        expect(decoded).toBe("A B");
    });
});