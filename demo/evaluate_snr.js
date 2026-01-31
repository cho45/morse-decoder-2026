/**
 * SNR Performance Evaluation Script for CW Decoder
 */
import ort from 'onnxruntime-node';
import { MorseGenerator, HFChannelSimulator } from './data_gen.js';
import { runFullInference, decodeFull, calculateCER, computeSpecFrame, computeSpecFrames, N_BINS, N_FFT, HOP_LENGTH, SAMPLE_RATE, LOOKAHEAD_FRAMES, SUBSAMPLING_RATE } from './inference.js';

// --- Configuration ---
const SNR_MIN = -20;
const SNR_MAX = 3;
const SNR_STEP = 1;
const SNR_RANGE = [];
for (let s = SNR_MIN; s <= SNR_MAX; s += SNR_STEP) {
    SNR_RANGE.push(s);
}
const SAMPLES_PER_SNR = 20;


function generateRandomText(length = 6) {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let result = "";
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result + " ";
}

async function evaluate() {
    const args = process.argv.slice(2);
    const modelIdx = args.indexOf('--model');
    const modelPath = modelIdx !== -1 ? args[modelIdx + 1] : 'demo/cw_decoder_quantized.onnx';

    console.log(`Loading model: ${modelPath}`);
    const session = await ort.InferenceSession.create(modelPath);
    const gen = new MorseGenerator(SAMPLE_RATE);
    const sim = new HFChannelSimulator(SAMPLE_RATE);

    console.log(`Starting SNR evaluation (${SAMPLES_PER_SNR} samples per SNR)...`);
    console.log(`SNR_2500(dB) | Avg CER | Raw Results`);
    console.log(`--------|---------|------------`);

    const mismatches = [];

    for (const snr of SNR_RANGE) {
        let totalDistance = 0;
        let totalChars = 0;
        const results = [];

        for (let i = 0; i < SAMPLES_PER_SNR; i++) {
            const targetText = generateRandomText(6);
            
            // Sync WPM estimation with Python visualize_snr_performance_onnx.py
            const sample_wpm = gen.estimateWpmForTargetFrames(
                targetText,
                1000, // target_frames (10s)
                15, 45 // min_wpm, max_wpm
            );

            const timing = gen.generateTiming(targetText, sample_wpm);
            const cleanWaveform = gen.generateWaveform(timing);
            
            // Python's data_gen.py always returns 10 seconds.
            // Pad JS waveform to match 10 seconds exactly.
            const targetSamples = 10.0 * SAMPLE_RATE;
            let fullWaveform = new Float32Array(targetSamples);
            fullWaveform.set(cleanWaveform.subarray(0, Math.min(cleanWaveform.length, targetSamples)));
            const noisyWaveform = sim.applyNoise(fullWaveform, { snr_2500: snr });
            const filteredWaveform = sim.applyFilter(noisyWaveform);

            // Normalize waveform to peak 1.0 (Sync with data_gen.py generate_sample)
            let maxVal = 0;
            for (let j = 0; j < filteredWaveform.length; j++) {
                const abs = Math.abs(filteredWaveform[j]);
                if (abs > maxVal) maxVal = abs;
            }
            if (maxVal > 0) {
                for (let j = 0; j < filteredWaveform.length; j++) {
                    filteredWaveform[j] /= maxVal;
                }
            }

            // Apply Lookahead Padding (Sync with visualize_snr_performance_onnx.py)
            const lookaheadSamples = LOOKAHEAD_FRAMES * HOP_LENGTH;
            const paddedWaveform = new Float32Array(filteredWaveform.length + lookaheadSamples);
            paddedWaveform.set(filteredWaveform);

            const specFrames = computeSpecFrames(paddedWaveform);

            // Inference (Chunked mode matching Python)
            const { logits, signal_logits, numClasses } = await runFullInference(session, specFrames, ort);
            
            // Trim back to original length (subsampled)
            // Sync with computeSpecFrames: nFrames = Math.floor((waveform.length - N_FFT) / HOP_LENGTH) + 1
            const originalSeqLen = Math.floor((filteredWaveform.length - N_FFT) / HOP_LENGTH) + 1;
            const originalOutLen = Math.floor((originalSeqLen + SUBSAMPLING_RATE - 1) / SUBSAMPLING_RATE);

            const trimmedLogits = logits.slice(0, originalOutLen * numClasses);
            const trimmedSigLogits = signal_logits.slice(0, originalOutLen * 4);

            const decodedText = decodeFull(trimmedLogits, trimmedSigLogits, numClasses);

            const cer = calculateCER(targetText, decodedText);
            totalDistance += cer * targetText.replace(/\s+/g, "").length; // Reconstruct total distance
            totalChars += targetText.replace(/\s+/g, "").length;
            results.push(cer);

            if (cer > 0 && snr >= -4) {
                mismatches.push({
                    snr,
                    targetText,
                    decodedText,
                    cer
                });
            }
        }

        const avgCer = totalDistance / totalChars;
        console.log(`${snr.toString().padStart(7)} | ${avgCer.toFixed(4)}  | ${results.map(r => r.toFixed(2)).join(' ')}`);
    }

    if (mismatches.length > 0) {
        console.log(`\n--- Mismatch Details (SNR >= -4dB) ---`);
        for (const m of mismatches) {
            console.log(`[SNR ${m.snr}dB] Mismatch:`);
            console.log(`  Ref: "${m.targetText}"`);
            console.log(`  Hyp: "${m.decodedText}"`);
            console.log(`  CER: ${m.cer.toFixed(4)}`);
        }
    }
}

evaluate().catch(err => {
    console.error(err);
    process.exit(1);
});