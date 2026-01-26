/**
 * CW Decoder - Inference & CTC Decoding Logic
 */

import { DSP } from './dsp.js';

// --- Constants (Sync with config.py / demo.js) ---
export const N_BINS = 14;
export const NUM_LAYERS = 4;
export const D_MODEL = 256;
export const N_HEAD = 4;
export const D_K = D_MODEL / N_HEAD;
export const KERNEL_SIZE = 31;

export const N_FFT = 512;
export const INPUT_LEN = 400;
export const HOP_LENGTH = 160;
export const F_MIN = 500.0;
export const SAMPLE_RATE = 16000;
export const LOOKAHEAD_FRAMES = 30;
export const SUBSAMPLING_RATE = 2;

const STD_CHARS = ",./0123456789?ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const PROSIGNS = ["<BT>", "<AR>", "<SK>", "<KA>"];
export const CHARS = [...STD_CHARS, ...PROSIGNS];
export const ID_TO_CHAR = {};
CHARS.forEach((char, i) => { ID_TO_CHAR[i + 1] = char; });

// --- Utilities ---

/**
 * STFT -> Cropped Spectrogram bin (Sync with demo.js applyFFTAndCrop)
 * @param {Float32Array} samples - Audio samples (INPUT_LEN or N_FFT)
 * @returns {Float32Array} Spectrogram bin [N_BINS]
 */
export function computeSpecFrame(samples) {
    const real = new Float32Array(N_FFT);
    const imag = new Float32Array(N_FFT);

    // Apply Hann window and copy to real buffer
    // Use N_FFT (512) instead of INPUT_LEN (400) to match torchaudio default win_length
    // Match PyTorch default (periodic=True) for STFT: denominator is N_FFT, not N_FFT - 1
    for (let i = 0; i < N_FFT; i++) {
        const val = samples[i] || 0;
        real[i] = val * (0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT)));
    }

    DSP.fft(real, imag);

    const binStart = Math.round(F_MIN * N_FFT / SAMPLE_RATE);
    const specFrame = new Float32Array(N_BINS);
    for (let i = 0; i < N_BINS; i++) {
        const k = binStart + i;
        const power = real[k] * real[k] + imag[k] * imag[k];
        // Match train.py/evaluate_detailed.py scaling: log1p(spec * 100) / 5.0
        specFrame[i] = Math.log1p(power * 100.0) / 5.0;
    }
    return specFrame;
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
 * Run inference on full spectrogram by chunking, matching Python's run_inference.
 * @param {object} session - ORT Inference Session
 * @param {Float32Array} specFrames - Flat array of spectrogram frames [T, N_BINS]
 * @param {object} ort - ONNX Runtime instance
 * @returns {Promise<object>} { logits, signal_logits }
 */
export async function runFullInference(session, specFrames, ort) {
    const seqLen = specFrames.length / N_BINS;
    const chunkSize = 40; // Standard chunk size (Sync with Python)
    let states = initStates(ort);

    const allLogits = [];
    const allSignalLogits = [];

    for (let i = 0; i < seqLen; i += chunkSize) {
        const end = Math.min(i + chunkSize, seqLen);
        const currentLen = end - i;
        
        // Ensure chunk size is multiple of 4 for subsampling (Sync with Python)
        const padLen = currentLen % 4 === 0 ? 0 : 4 - (currentLen % 4);
        const chunk = new Float32Array((currentLen + padLen) * N_BINS);
        chunk.set(specFrames.subarray(i * N_BINS, end * N_BINS));

        const x = new ort.Tensor('float32', chunk, [1, currentLen + padLen, N_BINS]);
        const inputs = { x, ...states };
        const results = await session.run(inputs);

        let logits = results.logits.data;
        let signalLogits = results.signal_logits.data;
        const numClasses = results.logits.dims[2];

        // Trim padding from logits of this chunk (Sync with Python run_inference)
        if (currentLen % 4 !== 0) {
            const validLen = Math.floor((currentLen + SUBSAMPLING_RATE - 1) / SUBSAMPLING_RATE);
            logits = logits.slice(0, validLen * numClasses);
            signalLogits = signalLogits.slice(0, validLen * 4);
        }

        allLogits.push(logits);
        allSignalLogits.push(signalLogits);

        // Update states
        states.sub_cache = results.new_sub_cache;
        for (let l = 0; l < NUM_LAYERS; l++) {
            states[`attn_k_${l}`] = results[`new_attn_k_${l}`];
            states[`attn_v_${l}`] = results[`new_attn_v_${l}`];
            states[`offset_${l}`] = results[`new_offset_${l}`];
            states[`conv_cache_${l}`] = results[`new_conv_cache_${l}`];
        }
        x.dispose();
    }

    // Concatenate all chunks
    const totalOutLen = allLogits.reduce((acc, l) => acc + l.length, 0) / (CHARS.length + 1);
    const fullLogits = new Float32Array(totalOutLen * (CHARS.length + 1));
    const fullSignalLogits = new Float32Array(totalOutLen * 4);

    let offsetLogits = 0;
    let offsetSig = 0;
    for (let i = 0; i < allLogits.length; i++) {
        fullLogits.set(allLogits[i], offsetLogits);
        fullSignalLogits.set(allSignalLogits[i], offsetSig);
        offsetLogits += allLogits[i].length;
        offsetSig += allSignalLogits[i].length;
    }

    // Final trim to expected length (Sync with Python run_inference)
    const expectedOutLen = Math.floor((seqLen + SUBSAMPLING_RATE - 1) / SUBSAMPLING_RATE);
    const trimmedLogits = fullLogits.slice(0, expectedOutLen * (CHARS.length + 1));
    const trimmedSignalLogits = fullSignalLogits.slice(0, expectedOutLen * 4);

    return {
        logits: trimmedLogits,
        signal_logits: trimmedSignalLogits,
        numClasses: CHARS.length + 1
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

    // 2. Space Reconstruction (Look back between peaks)
    let result = "";
    let lastPos = 0;
    for (let i = 0; i < decodedIndices.length; i++) {
        const idx = decodedIndices[i];
        const pos = decodedPositions[i];

        // Check for inter-word space (class 3) between last peak and current peak
        // Matching Python: any(sig_preds[last_pos:pos] == 3)
        let foundSpace = false;
        for (let t = lastPos; t < pos; t++) {
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

        if (foundSpace && result.length > 0 && result[result.length - 1] !== " ") {
            result += " ";
        }
        result += ID_TO_CHAR[idx] || "";
        lastPos = pos;
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