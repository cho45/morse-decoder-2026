import { describe, it, expect, beforeAll } from 'vitest';
import ort from 'onnxruntime-node';
import path from 'path';
import { initStates, runFullInference, runChunkInference, decodeFull, computeSpecFrames, calculateCER, N_BINS, CHARS, HOP_LENGTH, SAMPLE_RATE, LOOKAHEAD_FRAMES, ChunkedDecoder } from './inference.js';
import { MorseGenerator } from './data_gen.js';

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
        // The model file (cw_decoder_quantized.onnx) has 63 classes.
        expect(numClasses).toBe(63);
        
        // Subsampling rate is 2, so 8 frames -> 4 output frames
        expect(logits.length / numClasses).toBe(4);
    });

    it('should run chunk inference and update states cumulatively', async () => {
        let states = initStates(ort);
        const chunkSize = 12; // Must be multiple of 4 (ONNX Runtime requirement)
        const chunk = new Float32Array(chunkSize * N_BINS).fill(0.1);

        // --- First Inference ---
        const result1 = await runChunkInference(session, chunk, states, ort);

        expect(result1).toHaveProperty('logits');
        expect(result1).toHaveProperty('nextStates');

        // Subsampling rate is 2, so 12 frames -> 6 output frames
        const outFrames1 = chunkSize / 2;

        expect(result1.nextStates.offset_0.data[0]).toBe(BigInt(outFrames1));
        // Attention cache should grow by outFrames1
        expect(result1.nextStates.attn_k_0.dims[2]).toBe(outFrames1);

        // --- Second Inference ---
        const result2 = await runChunkInference(session, chunk, result1.nextStates, ort);

        // Expected cumulative output frames (6 + 6 = 12)
        const totalOutFrames = outFrames1 + outFrames1;

        expect(result2.nextStates.offset_0.data[0]).toBe(BigInt(totalOutFrames));
        // Attention cache should grow to totalOutFrames
        expect(result2.nextStates.attn_k_0.dims[2]).toBe(totalOutFrames);

        // Ensure PCEN state is different from first inference result
        expect(result2.nextStates.pcen_state.data).not.toEqual(result1.nextStates.pcen_state.data);
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

    it('should insert space after character, not before (space timing test)', () => {
        /**
         * Test space insertion timing with realistic CW signal timeline.
         *
         * Timeline:
         * - t=2: 'H' fires (CTC) after H signal ends
         * - t=4: 'I' fires (CTC) after I signal ends
         * - t=7-9: word space signal detected (sig_logits class 3)
         * - t=12: 'T' fires (CTC) after T signal ends
         *
         * Expected: "HI T" where space is inserted after 'I', not before 'T'.
         * This reflects that the space logically belongs to the word "HI", not to "T".
         */
        const numClasses = CHARS.length + 1;
        const T = 15;
        const allLogits = new Float32Array(T * numClasses).fill(-10);
        const allSignalLogits = new Float32Array(T * 4).fill(-10);

        // Set background signal (class 0) as default
        for (let t = 0; t < T; t++) {
            allSignalLogits[t * 4 + 0] = 2.0;
        }

        // CTC spikes
        const hIndex = CHARS.indexOf('H') + 1;
        const iIndex = CHARS.indexOf('I') + 1;
        const tIndex = CHARS.indexOf('T') + 1;

        allLogits[2 * numClasses + hIndex] = 10.0;  // 'H' at t=2
        allLogits[4 * numClasses + iIndex] = 10.0;  // 'I' at t=4
        allLogits[12 * numClasses + tIndex] = 10.0; // 'T' at t=12

        // Word space signal at t=7-9
        allSignalLogits[7 * 4 + 3] = 10.0;
        allSignalLogits[8 * 4 + 3] = 10.0;
        allSignalLogits[9 * 4 + 3] = 10.0;

        const decoded = decodeFull(allLogits, allSignalLogits, numClasses);

        // Space should be inserted after 'I', creating "HI T"
        expect(decoded).toBe("HI T");
    });

    it('should handle multiple words correctly', () => {
        /**
         * Test with multiple words: "CQ DE K"
         *
         * Timeline:
         * - t=1: 'C', t=3: 'Q' -> space at t=5-6
         * - t=8: 'D', t=10: 'E' -> space at t=12-13
         * - t=15: 'K'
         */
        const numClasses = CHARS.length + 1;
        const T = 20;
        const allLogits = new Float32Array(T * numClasses).fill(-10);
        const allSignalLogits = new Float32Array(T * 4).fill(-10);

        // Set background signal
        for (let t = 0; t < T; t++) {
            allSignalLogits[t * 4 + 0] = 2.0;
        }

        // CTC spikes
        const cIndex = CHARS.indexOf('C') + 1;
        const qIndex = CHARS.indexOf('Q') + 1;
        const dIndex = CHARS.indexOf('D') + 1;
        const eIndex = CHARS.indexOf('E') + 1;
        const kIndex = CHARS.indexOf('K') + 1;

        allLogits[1 * numClasses + cIndex] = 10.0;
        allLogits[3 * numClasses + qIndex] = 10.0;
        allLogits[8 * numClasses + dIndex] = 10.0;
        allLogits[10 * numClasses + eIndex] = 10.0;
        allLogits[15 * numClasses + kIndex] = 10.0;

        // Word spaces
        allSignalLogits[5 * 4 + 3] = 10.0;
        allSignalLogits[6 * 4 + 3] = 10.0;
        allSignalLogits[12 * 4 + 3] = 10.0;
        allSignalLogits[13 * 4 + 3] = 10.0;

        const decoded = decodeFull(allLogits, allSignalLogits, numClasses);
        expect(decoded).toBe("CQ DE K");
    });

    it('should decode "CQ" correctly using both full and chunk inference', async () => {
        const gen = new MorseGenerator(SAMPLE_RATE);
        const targetText = "CQ CQ DE JH1UMV K";
        const timing = gen.generateTiming(targetText, 25);
        const cleanWaveform = gen.generateWaveform(timing);
        
        // Add lookahead padding
        const lookaheadSamples = LOOKAHEAD_FRAMES * HOP_LENGTH;
        const paddedWaveform = new Float32Array(cleanWaveform.length + lookaheadSamples);
        paddedWaveform.set(cleanWaveform);
        
        const specFrames = computeSpecFrames(paddedWaveform);
        const seqLen = specFrames.length / N_BINS;

        // 1. Full Inference
        const fullResult = await runFullInference(session, specFrames, ort);
        const fullDecoded = decodeFull(fullResult.logits, fullResult.signal_logits, fullResult.numClasses);
        // Use CER to be robust against minor spacing issues, but ensure content is correct
        expect(calculateCER(targetText, fullDecoded)).toBe(0);

        // 2. Chunk Inference (Streaming)
        let states = initStates(ort);
        const chunkSize = 40; // Must be multiple of 4 (ONNX Runtime requirement)
        const allLogits = [];
        const allSignalLogits = [];
        let numClasses = 0;

        for (let i = 0; i < seqLen; i += chunkSize) {
            const end = Math.min(i + chunkSize, seqLen);
            const chunk = specFrames.slice(i * N_BINS, end * N_BINS);
            const result = await runChunkInference(session, chunk, states, ort);
            allLogits.push(result.logits);
            allSignalLogits.push(result.signalLogits);
            states = result.nextStates;
            numClasses = result.numClasses;
        }

        const totalOutFrames = allLogits.reduce((acc, l) => acc + l.length / numClasses, 0);
        const mergedLogits = new Float32Array(totalOutFrames * numClasses);
        const mergedSignalLogits = new Float32Array(totalOutFrames * 4);
        let offsetLogits = 0;
        let offsetSig = 0;
        for (let i = 0; i < allLogits.length; i++) {
            mergedLogits.set(allLogits[i], offsetLogits);
            mergedSignalLogits.set(allSignalLogits[i], offsetSig);
            offsetLogits += allLogits[i].length;
            offsetSig += allSignalLogits[i].length;
        }

        const chunkDecoded = decodeFull(mergedLogits, mergedSignalLogits, numClasses);
        expect(calculateCER(targetText, chunkDecoded)).toBe(0);
    });

    describe('ChunkedDecoder', () => {
        it('should decode streaming frames correctly with space after character', () => {
            const numClasses = CHARS.length + 1;
            const decoder = new ChunkedDecoder(numClasses);

            // Frame 0: 'H' fires
            let ctcLogits0 = new Float32Array(numClasses).fill(-10);
            let sigLogits0 = new Float32Array(4).fill(-10);
            const hIndex = CHARS.indexOf('H') + 1;
            ctcLogits0[hIndex] = 10.0;
            sigLogits0[0] = 2.0; // background

            let result = decoder.decodeFrame(ctcLogits0, sigLogits0, 0.5);
            expect(result.text).toBe('H');
            expect(result.newChar).toBe('H');

            // Frame 1-2: background
            for (let i = 1; i < 3; i++) {
                let ctcLogits = new Float32Array(numClasses).fill(-10);
                let sigLogits = new Float32Array(4).fill(-10);
                ctcLogits[0] = 10.0; // blank
                sigLogits[0] = 2.0;
                result = decoder.decodeFrame(ctcLogits, sigLogits, 0.5);
            }

            // Frame 3: 'I' fires
            let ctcLogits3 = new Float32Array(numClasses).fill(-10);
            let sigLogits3 = new Float32Array(4).fill(-10);
            const iIndex = CHARS.indexOf('I') + 1;
            ctcLogits3[iIndex] = 10.0;
            sigLogits3[0] = 2.0;

            result = decoder.decodeFrame(ctcLogits3, sigLogits3, 0.5);
            expect(result.text).toBe('HI');
            expect(result.newChar).toBe('I');

            // Frame 4-6: word space detected
            for (let i = 4; i < 7; i++) {
                let ctcLogits = new Float32Array(numClasses).fill(-10);
                let sigLogits = new Float32Array(4).fill(-10);
                ctcLogits[0] = 10.0; // blank
                sigLogits[3] = 10.0; // word space
                result = decoder.decodeFrame(ctcLogits, sigLogits, 0.5);
            }

            // Frame 7: 'T' fires (space should be inserted before this character)
            let ctcLogits7 = new Float32Array(numClasses).fill(-10);
            let sigLogits7 = new Float32Array(4).fill(-10);
            const tIndex = CHARS.indexOf('T') + 1;
            ctcLogits7[tIndex] = 10.0;
            sigLogits7[0] = 2.0;

            result = decoder.decodeFrame(ctcLogits7, sigLogits7, 0.5);
            expect(result.text).toBe('HI T');
            expect(result.newChar).toBe('T');
            expect(result.spaceInserted).toBe(true);
        });

        it('should reset state correctly', () => {
            const decoder = new ChunkedDecoder(63);

            let ctcLogits = new Float32Array(63).fill(-10);
            let sigLogits = new Float32Array(4).fill(-10);
            ctcLogits[1] = 10.0; // some character
            sigLogits[0] = 2.0;

            decoder.decodeFrame(ctcLogits, sigLogits, 0.5);
            expect(decoder.getText()).not.toBe('');

            decoder.reset();
            expect(decoder.getText()).toBe('');
        });
    });
});