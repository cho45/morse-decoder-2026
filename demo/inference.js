/**
 * CW Decoder - Inference & CTC Decoding Logic
 */

import { DSP } from './dsp.js';

// --- Constants (Sync with config.py / demo.js) ---
export const N_BINS = 14;
export const NUM_LAYERS = 4; // Check if this should be 6 based on recent config.py changes
export const D_MODEL = 256;
export const N_HEAD = 4;
export const D_K = D_MODEL / N_HEAD;
export const KERNEL_SIZE = 31;

export const N_FFT = 512;
export const HOP_LENGTH = 160;
export const F_MIN = 500.0;
export const SAMPLE_RATE = 16000;
export const LOOKAHEAD_FRAMES = 30;
export const SUBSAMPLING_RATE = 2;

// Vocabulary MUST match config.py and model dimensions (63 classes = 62 chars + 1 blank)
export const CHARS = ["!", "\"", "$", "&", "'", "(", ")", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "_", "<NJ>", "<DDD>", "<SK>", "<KA>", "<SOS>", "<VE>", "<HH>", "<AA>"];
export const ID_TO_CHAR = {};
CHARS.forEach((char, i) => { ID_TO_CHAR[i + 1] = char; });
console.log("Vocabulary initialized:", CHARS.length, "chars");

// --- Utilities ---

/**
 * STFT -> Cropped Spectrogram bin (Sync with demo.js applyFFTAndCrop)
 * @param {Float32Array} samples - Audio samples (INPUT_LEN or N_FFT)
 * @param {number} sampleRate - Current audio sample rate (default: 16000)
 * @returns {Float32Array} Spectrogram bin [N_BINS]
 */
export function computeSpecFrame(samples, sampleRate = SAMPLE_RATE) {
    const real = new Float32Array(N_FFT);
    const imag = new Float32Array(N_FFT);

    // Apply Hann window and copy to real buffer
    for (let i = 0; i < N_FFT; i++) {
        const val = samples[i] || 0;
        real[i] = val * (0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT)));
    }

    DSP.fft(real, imag);

    const binStart = Math.round(F_MIN * N_FFT / sampleRate);
    const specFrame = new Float32Array(N_BINS);
    for (let i = 0; i < N_BINS; i++) {
        const k = binStart + i;
        const power = real[k] * real[k] + imag[k] * imag[k];
        // PCEN handles log scaling and normalization inside the model.
        // We just pass the raw power (spectrogram) values.
        specFrame[i] = power;
    }
    return specFrame;
}

/**
 * Compute all spectrogram frames for a given waveform.
 * @param {Float32Array} waveform - Audio samples
 * @param {number} sampleRate - Current audio sample rate
 * @returns {Float32Array} Flat array of spectrogram frames [T, N_BINS]
 */
export function computeSpecFrames(waveform, sampleRate = SAMPLE_RATE) {
    const nFrames = Math.floor((waveform.length - N_FFT) / HOP_LENGTH) + 1;
    const specFrames = new Float32Array(nFrames * N_BINS);
    for (let i = 0; i < nFrames; i++) {
        const start = i * HOP_LENGTH;
        let frame = waveform.slice(start, start + N_FFT);
        if (frame.length < N_FFT) {
            const padded = new Float32Array(N_FFT);
            padded.set(frame);
            frame = padded;
        }
        const spec = computeSpecFrame(frame, sampleRate);
        specFrames.set(spec, i * N_BINS);
    }
    return specFrames;
}

export function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

/**
 * Initialize inference states for the streaming model.
 * @param {object} ort - ONNX Runtime instance
 * @returns {object} Initial states
 */
export function initStates(ort) {
    const states = {
        // [FIX] PCEN state should be initialized with 0s, but the model logic handles init.
        // However, to match training behavior where state=None implies curr_state=x[0],
        // we might need to handle the first frame specially if the model expects it.
        // Current model.py logic: if state is provided, use it. If not, use x[0] (or 0s if T=0).
        // Since we always provide state here (zeros), the model uses these zeros as the previous EMA.
        // This causes a "warmup" period where EMA starts from 0 instead of x[0].
        // Ideally, we should detect the first call and pass something to indicate "init with x[0]",
        // but ONNX requires tensor inputs.
        // A workaround is to not change this here, but be aware of the warmup.
        // Or, we could pre-calculate the first PCEN state if we had access to the first frame here.
        // For now, we keep it as zeros, acknowledging the slight discrepancy at start.
        pcen_state: new ort.Tensor('float32', new Float32Array(1 * 1 * N_BINS), [1, 1, N_BINS]),
        sub_cache: new ort.Tensor('float32', new Float32Array(1 * 1 * 2 * N_BINS), [1, 1, 2, N_BINS])
    };
    for (let i = 0; i < NUM_LAYERS; i++) {
        states[`attn_k_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, N_HEAD, 0, D_K]);
        states[`attn_v_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, N_HEAD, 0, D_K]);
        states[`offset_${i}`] = new ort.Tensor('int64', new BigInt64Array([0n]), []);
        states[`conv_cache_${i}`] = new ort.Tensor('float32', new Float32Array(1 * D_MODEL * (KERNEL_SIZE - 1)), [1, D_MODEL, KERNEL_SIZE - 1]);
    }
    return states;
}

/**
 * Run inference on a single chunk of spectrogram frames.
 * Handles state updates and padding internally.
 * @param {object} session - ORT Inference Session
 * @param {Float32Array} chunkFrames - Spectrogram frames [T, N_BINS]
 * @param {object} states - Current inference states
 * @param {object} ort - ONNX Runtime instance
 * @returns {Promise<object>} { logits, signal_logits, boundary_logits, nextStates }
 */
let inferenceCount = 0;

export async function runChunkInference(session, chunkFrames, states, ort) {
    inferenceCount++;
    const currentLen = chunkFrames.length / N_BINS;

    if (currentLen % 4 !== 0) {
        throw new Error(
            `Input chunk length must be a multiple of 4, got ${currentLen}. ` +
            'This is an ONNX Runtime limitation: state updates happen before output is returned, ' +
            'so padding would corrupt the streaming model cache (ConvSubsampling, ConformerConvModule, Attention). ' +
            'The Python model itself supports any chunk size, but ONNX Runtime requires 4-multiple chunks.'
        );
    }

    const tensorStart = performance.now();
    const x = new ort.Tensor('float32', chunkFrames, [1, currentLen, N_BINS]);
    const tensorTime = performance.now() - tensorStart;

    const inputs = { x, ...states };
    const sessionStart = performance.now();
    let results;
    try {
        results = await session.run(inputs);
    } catch (e) {
        console.error(`session.run failed at inference #${inferenceCount}`);
        throw e;
    }
    const sessionTime = performance.now() - sessionStart;

    console.log(`ONNX Runtime session.run #${inferenceCount}: ${sessionTime.toFixed(2)}ms`);

    const numClasses = results.logits.dims[2];
    const actualOutFrames = results.logits.dims[1];

    // Expected output frames after subsampling (ceil division)
    const expectedOutFrames = Math.floor((currentLen + SUBSAMPLING_RATE - 1) / SUBSAMPLING_RATE);

    // Copy data before dispose (tensor.data is a view into WASM memory)
    let logits = new Float32Array(results.logits.data);
    let signalLogits = new Float32Array(results.signal_logits.data);
    let boundaryLogits = new Float32Array(results.boundary_logits.data);

    // Sanity check: output should match expected length
    if (actualOutFrames > expectedOutFrames) {
        logits = logits.slice(0, expectedOutFrames * numClasses);
        signalLogits = signalLogits.slice(0, expectedOutFrames * 4);
        boundaryLogits = boundaryLogits.slice(0, expectedOutFrames);
    }

    const nextStates = {};
    nextStates.pcen_state = results.new_pcen_state;
    nextStates.sub_cache = results.new_sub_cache;
    for (let l = 0; l < NUM_LAYERS; l++) {
        nextStates[`attn_k_${l}`] = results[`new_attn_k_${l}`];
        nextStates[`attn_v_${l}`] = results[`new_attn_v_${l}`];
        nextStates[`offset_${l}`] = results[`new_offset_${l}`];
        nextStates[`conv_cache_${l}`] = results[`new_conv_cache_${l}`];
    }

    // Clean up temporary tensors (keep nextStates)
    x.dispose();

    // Dispose output tensors after extracting data (logits/signalLogits/boundaryLogits are Float32Array copies)
    results.logits.dispose();
    results.signal_logits.dispose();
    results.boundary_logits.dispose();

    return {
        logits,
        signalLogits,
        boundaryLogits,
        nextStates,
        numClasses
    };
}

/**
 * Run inference on full spectrogram by chunking, matching Python's run_inference.
 * @param {object} session - ORT Inference Session
 * @param {Float32Array} specFrames - Flat array of spectrogram frames [T, N_BINS]
 * @param {object} ort - ONNX Runtime instance
 * @returns {Promise<object>} { logits, signal_logits }
 */
export async function runFullInference(session, specFrames, ort) {
    let seqLen = specFrames.length / N_BINS;
    const chunkSize = 40; // Standard chunk size (Sync with Python)

    // Ensure total length is a multiple of 4 for ONNX Runtime streaming compatibility
    if (seqLen % 4 !== 0) {
        const paddedLen = Math.ceil(seqLen / 4) * 4;
        const paddedSpec = new Float32Array(paddedLen * N_BINS);
        paddedSpec.set(specFrames);
        specFrames = paddedSpec;
        seqLen = paddedLen;
    }

    let states = initStates(ort);

    const allLogits = [];
    const allSignalLogits = [];
    let totalOutFrames = 0;
    let numClasses = 0;

    for (let i = 0; i < seqLen; i += chunkSize) {
        const end = Math.min(i + chunkSize, seqLen);
        const currentLen = end - i;
        const chunk = specFrames.subarray(i * N_BINS, end * N_BINS);

        const result = await runChunkInference(session, chunk, states, ort);
        
        allLogits.push(result.logits);
        allSignalLogits.push(result.signalLogits);
        states = result.nextStates;
        totalOutFrames += result.logits.length / result.numClasses;
        numClasses = result.numClasses;
    }

    // Concatenate all chunks
    const fullLogits = new Float32Array(totalOutFrames * numClasses);
    const fullSignalLogits = new Float32Array(totalOutFrames * 4);

    let offsetLogits = 0;
    let offsetSig = 0;
    for (let i = 0; i < allLogits.length; i++) {
        fullLogits.set(allLogits[i], offsetLogits);
        fullSignalLogits.set(allSignalLogits[i], offsetSig);
        offsetLogits += allLogits[i].length;
        offsetSig += allSignalLogits[i].length;
    }

    return {
        logits: fullLogits,
        signal_logits: fullSignalLogits,
        numClasses: numClasses
    };
}

/**
 * CTC Decoding with Space Reconstruction.
 * @param {Float32Array} allLogits - CTC logits for all frames [T, CHARS.length + 1]
 * @param {Float32Array} allSignalLogits - Signal logits for all frames [T, 4]
 * @param {number} numClasses - Number of CTC classes
 * @returns {string} Decoded text
 */
export function decodeFull(allLogits, allSignalLogits, numClasses) {
    const T = allLogits.length / numClasses;
    
    // 1. CTC Greedy Decoding (Find peaks and their positions)
    const decodedIndices = [];
    const decodedPositions = [];
    let prevId = -1;

    for (let t = 0; t < T; t++) {
        const logits = allLogits.slice(t * numClasses, (t + 1) * numClasses);
        let maxId = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxId = i;
            }
        }

        // Standard CTC Greedy: Skip blank (0) and repeated characters
        if (maxId !== 0 && maxId !== prevId) {
            decodedIndices.push(maxId);
            decodedPositions.push(t);
        }
        prevId = maxId;
    }

    // 2. Gated Space Reconstruction
    let result = "";

    for (let i = 0; i < decodedIndices.length; i++) {
        const idx = decodedIndices[i];
        const pos = decodedPositions[i];

        // Append character first
        result += ID_TO_CHAR[idx] || "";

        // Check for inter-word space AFTER current character
        // Space is detected between current position and next character position
        // This reflects the CW signal timing: character fires → space detected → next character fires
        if (i + 1 < decodedPositions.length) {
            const nextPos = decodedPositions[i + 1];
            // Check if word space (class 3) exists between current and next character
            let foundSpace = false;
            for (let t = pos; t < nextPos; t++) {
                const sigLogits = allSignalLogits.slice(t * 4, (t + 1) * 4);
                // Argmax
                let maxSigId = 0;
                let maxSigVal = -Infinity;
                for (let s = 0; s < 4; s++) {
                    if (sigLogits[s] > maxSigVal) {
                        maxSigVal = sigLogits[s];
                        maxSigId = s;
                    }
                }
                if (maxSigId === 3) {
                    foundSpace = true;
                    break;
                }
            }

            if (foundSpace) {
                result += " ";
            }
        }
    }

    return result.trim();
}

/**
 * Calculate Character Error Rate matching Python's visualize_snr_performance.py
 * @param {string} ref - Reference text
 * @param {string} hyp - Hypothesis text
 * @returns {number} CER
 */
export function calculateCER(ref, hyp) {
    const PROSIGNS = ["<BT>", "<AR>", "<SK>", "<KA>"];
    // Prosign mapping to single characters for fair evaluation
    const prosignMapping = {};
    PROSIGNS.forEach((ps, i) => {
        prosignMapping[ps] = String.fromCharCode(i + 1);
    });

    function mapProsigns(text) {
        let res = text.replace(/\s+/g, "");
        for (const [ps, char] of Object.entries(prosignMapping)) {
            // Replace all occurrences
            res = res.split(ps).join(char);
        }
        return res;
    }

    const refMapped = mapProsigns(ref);
    const hypMapped = mapProsigns(hyp);

    if (!refMapped) {
        return hypMapped ? 1.0 : 0.0;
    }

    const n = refMapped.length;
    const m = hypMapped.length;
    const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));

    for (let i = 0; i <= n; i++) dp[i][0] = i;
    for (let j = 0; j <= m; j++) dp[0][j] = j;

    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
            const cost = refMapped[i - 1] === hypMapped[j - 1] ? 0 : 1;
            dp[i][j] = Math.min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            );
        }
    }

    return dp[n][m] / n;
}

/**
 * ChunkedDecoder - Stateful streaming decoder for real-time CW decoding.
 * Handles frame-by-frame CTC decoding with space reconstruction.
 */
export class ChunkedDecoder {
    /**
     * @param {number} numClasses - Number of CTC classes (including blank)
     */
    constructor(numClasses) {
        this.numClasses = numClasses;
        this.reset();
    }

    /**
     * Reset decoder state
     */
    reset() {
        this.decodedText = "";
        this.lastCharId = -1;
        this.spaceAfterLastChar = false; // Track if word space was detected after last character
    }

    /**
     * Decode a single frame
     * @param {Float32Array} ctcLogits - CTC logits for this frame [numClasses]
     * @param {Float32Array} sigLogits - Signal logits for this frame [4]
     * @param {number} boundThreshold - Boundary probability threshold (unused in basic version)
     * @returns {object} { text, newChar, spaceInserted }
     */
    decodeFrame(ctcLogits, sigLogits, boundThreshold = 0.5) {
        // CTC Greedy Decoding
        let maxId = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < ctcLogits.length; i++) {
            if (ctcLogits[i] > maxVal) {
                maxVal = ctcLogits[i];
                maxId = i;
            }
        }

        // Signal Argmax
        let maxSigId = 0;
        let maxSigVal = -Infinity;
        for (let s = 0; s < 4; s++) {
            if (sigLogits[s] > maxSigVal) {
                maxSigVal = sigLogits[s];
                maxSigId = s;
            }
        }

        let newChar = null;
        let spaceInserted = false;

        // Check for word space signal (class 3)
        if (maxSigId === 3 && this.decodedText.length > 0) {
            this.spaceAfterLastChar = true;
        }

        // Character emission: Skip blank (0) and repeated characters
        if (maxId !== 0 && maxId !== this.lastCharId) {
            // Insert space before new character if space was detected after previous character
            if (this.spaceAfterLastChar) {
                this.decodedText += " ";
                this.spaceAfterLastChar = false;
                spaceInserted = true;
            }

            const char = ID_TO_CHAR[maxId] || "";
            this.decodedText += char;
            newChar = char;
        }

        this.lastCharId = maxId;

        return {
            text: this.decodedText,
            newChar: newChar,
            spaceInserted: spaceInserted
        };
    }

    /**
     * Get current decoded text
     * @returns {string}
     */
    getText() {
        return this.decodedText;
    }
}