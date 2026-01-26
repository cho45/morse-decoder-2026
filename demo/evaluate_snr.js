/**
 * SNR Performance Evaluation Script for CW Decoder
 */
import ort from 'onnxruntime-node';
import { MorseGenerator, HFChannelSimulator } from './data_gen.js';
import { runFullInference, decodeFull, calculateCER, computeSpecFrame, N_BINS, N_FFT, HOP_LENGTH, SAMPLE_RATE, LOOKAHEAD_FRAMES, SUBSAMPLING_RATE } from './inference.js';

// --- Configuration ---
const SNR_RANGE = [-18, -16, -14, -12, -10, -8, -6, -4, -2];
const SAMPLES_PER_SNR = 10;
const TEST_PHRASES = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "CQ CQ CQ DE K7ABC K",
    "UR RST IS 599 599 BT OP IS JOHN BT QTH IS SEATTLE WA",
    "PARIS PARIS PARIS",
    "12345 67890",
    "KILO CODE EVALUATION",
    "WEATHER IS FINE",
    "73 ES GL",
    "MORSE CODE DECODER TEST",
    "SOS SOS SOS"
];


/**
 * Waveform -> Spectrogram frames
 */
/**
 * Waveform -> Spectrogram frames
 */
function computeSpecFrames(waveform) {
    const nFrames = Math.floor((waveform.length - N_FFT) / HOP_LENGTH) + 1;
    const specFrames = new Float32Array(nFrames * N_BINS);
    for (let i = 0; i < nFrames; i++) {
        const frame = waveform.slice(i * HOP_LENGTH, i * HOP_LENGTH + N_FFT);
        const spec = computeSpecFrame(frame);
        specFrames.set(spec, i * N_BINS);
    }
    return specFrames;
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
    console.log(`SNR(dB) | Avg CER | Raw Results`);
    console.log(`--------|---------|------------`);

    for (const snr of SNR_RANGE) {
        let totalDistance = 0;
        let totalChars = 0;
        const results = [];

        for (let i = 0; i < SAMPLES_PER_SNR; i++) {
            const targetText = TEST_PHRASES[i % TEST_PHRASES.length];
            const timing = gen.generateTiming(targetText, 20);
            const cleanWaveform = gen.generateWaveform(timing);
            const noisyWaveform = sim.applyNoise(cleanWaveform, snr);
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
            const originalSeqLen = Math.floor((filteredWaveform.length + HOP_LENGTH - 1) / HOP_LENGTH);
            const originalOutLen = Math.floor((originalSeqLen + SUBSAMPLING_RATE - 1) / SUBSAMPLING_RATE);

            const trimmedLogits = logits.slice(0, originalOutLen * numClasses);
            const trimmedSigLogits = signal_logits.slice(0, originalOutLen * 4);

            const decodedText = decodeFull(trimmedLogits, trimmedSigLogits, numClasses);

            const cer = calculateCER(targetText, decodedText);
            totalDistance += cer * targetText.replace(/\s+/g, "").length; // Reconstruct total distance
            totalChars += targetText.replace(/\s+/g, "").length;
            results.push(cer);

            if (cer > 0 && snr >= -4) {
                console.log(`[SNR ${snr}dB] Mismatch:`);
                console.log(`  Ref: "${targetText}"`);
                console.log(`  Hyp: "${decodedText}"`);
                console.log(`  CER: ${cer.toFixed(4)}`);
            }
        }

        const avgCer = totalDistance / totalChars;
        console.log(`${snr.toString().padStart(7)} | ${avgCer.toFixed(4)}  | ${results.map(r => r.toFixed(2)).join(' ')}`);
    }
}

evaluate().catch(err => {
    console.error(err);
    process.exit(1);
});