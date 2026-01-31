/**
 * CW Decoder - Main Demo Script
 */
import {
    N_FFT, HOP_LENGTH, N_BINS,
    CHARS,
    computeSpecFrame
} from './inference.js';
import { StreamInference } from './stream-inference.js';
import { MORSE_DICT } from './data_gen.js';
import { getViridisColor, powerToDBNormalized, normalizeDB } from './visualization.js';

// ort.env.webgpu.profiling = { mode: 'default' };

// --- Constants & Config ---
const SAMPLE_RATE = 16000; // Model expected sample rate

// --- State Variables ---
let audioContext = null;
let morseNode = null;
let oscillator = null;
let gainNode = null;
let masterGainNode = null;
let noiseNode = null;
let filterNode = null;
let outputFilterNode = null;
let session = null;
let isRunning = false;
let streamInference = null; // StreamInference instance
let audioBuffer = new Float32Array(N_FFT); // Sliding window for FFT
let morseAbortController = null;
let skipCount = 0;
let useWebGPU = false;

// Visualization Buffers (History) - melHistory is local, others come from StreamInference
const HISTORY_LEN = 200; // Number of frames to show
let melHistory = [];

// Canvas Contexts
const melCanvas = document.getElementById('melCanvas');
const ctcCanvas = document.getElementById('ctcCanvas');
const sigCanvas = document.getElementById('sigCanvas');
const gapCanvas = document.getElementById('gapCanvas');
const boundCanvas = document.getElementById('boundCanvas');

// --- UI Elements ---
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const modelSelect = document.getElementById('modelSelect');
const inputText = document.getElementById('inputText');
const output = document.getElementById('output');
const status = document.getElementById('status');
const debugInfo = document.getElementById('debugInfo');

const wpmSlider = document.getElementById('wpm');
const freqSlider = document.getElementById('frequency');
const snrSlider = document.getElementById('snr');
const jitterSlider = document.getElementById('jitter');
const volumeSlider = document.getElementById('volume');

// Update value displays
const updateSliders = () => {
    document.getElementById('wpmValue').textContent = wpmSlider.value;
    document.getElementById('freqValue').textContent = freqSlider.value;
    document.getElementById('snrValue').textContent = snrSlider.value;
    document.getElementById('jitterValue').textContent = jitterSlider.value;
    document.getElementById('volumeValue').textContent = volumeSlider.value;
};
[wpmSlider, freqSlider, snrSlider, jitterSlider, volumeSlider].forEach(s => {
    s.oninput = () => {
        updateSliders();
        if (s.id === 'volume' && masterGainNode) {
            masterGainNode.gain.setTargetAtTime(parseFloat(volumeSlider.value), audioContext.currentTime, 0.01);
        }
    };
});
updateSliders();

// --- ONNX Inference ---
async function initONNX() {
    try {
        const modelPath = modelSelect.value;
        status.textContent = `モデル読み込み中: ${modelPath.split('/').pop()}...`;

        // Dispose existing StreamInference
        if (streamInference) {
            streamInference.dispose();
            streamInference = null;
        }

        // 既存のセッションがあれば破棄
        if (session) {
            session = null;
        }

        // URLハッシュでプロバイダーを切り替え (#webgpu または #wasm)
        useWebGPU = window.location.hash === '#webgpu';
        console.log('URL hash:', window.location.hash);
        console.log('Use WebGPU:', useWebGPU);
        const providers = useWebGPU ? ['webgpu'] : ['wasm'];

        console.log(`Loading model with providers: ${providers}`);
        console.log(`Model path: ${modelPath}`);

        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: providers,
            graphOptimizationLevel: 'all',
            executionMode: 'sequential'
        });

        // Create StreamInference instance
        streamInference = new StreamInference(session, ort, {
            chunkSize: 12,
            useWebGPU: useWebGPU,
            historyLength: HISTORY_LEN
        });

        // Set up event listeners
        streamInference.addEventListener('result', (e) => {
            output.textContent = e.detail.text;
        });

        streamInference.addEventListener('frame', () => {
            drawVisualizations();
            updateDebugInfo();
        });

        status.textContent = `準備完了 (${useWebGPU ? 'webgpu' : 'wasm'})`;
        debugInfo.innerHTML = `モデル: 読み込み完了 (${useWebGPU ? 'webgpu' : 'wasm'})<br>モード切替: URLに #webgpu または #wasm を追加<br>スキップ数: 0`;
    } catch (e) {
        status.textContent = "モデル読み込み失敗: " + e;
        console.error(e);
    }
}

// StreamInference handles all inference state management internally.
// We only need to push frames and listen for events.

function updateDebugInfo() {
    if (!session) return;

    const frameCount = streamInference ? streamInference.frameCount : 0;

    debugInfo.innerHTML = `モデル: ${isRunning ? '動作中' : '読み込み完了'} (${useWebGPU ? 'webgpu' : 'wasm'})<br>` +
                         `処理フレーム数: ${frameCount}<br>` +
                         `スキップ数: ${skipCount}`;
}


function drawVisualizations() {
    drawMel();
    // Get history from StreamInference
    const ctcHistory = streamInference ? streamInference.getCTCHistory() : [];
    const sigHistory = streamInference ? streamInference.getSignalHistory().map(h => h.probs) : [];
    const boundHistory = streamInference ? streamInference.getBoundaryHistory() : [];
    const decodedEvents = streamInference ? streamInference.getEvents() : [];
    const totalFrames = streamInference ? streamInference.frameCount : 0;

    drawProbs(ctcCanvas, ctcHistory, ["blank", ...CHARS], true, decodedEvents, totalFrames);

    // Split signal history: Dit/Dah (indices 1,2) and Space/Inter-Word (indices 0,3)
    const ditDahHistory = sigHistory.map(probs => [probs[1], probs[2]]);
    const gapHistory = sigHistory.map(probs => [probs[0], probs[3]]);
    drawProbs(sigCanvas, ditDahHistory, ["Dit", "Dah"], false, [], 0);
    drawProbs(gapCanvas, gapHistory, ["Space", "Inter-Word"], false, [], 0);

    drawBoundary(boundHistory);
}

function drawMel() {
    const ctx = melCanvas.getContext('2d');
    const w = melCanvas.width;
    const h = melCanvas.height;
    // Dark background for Viridis colormap
    ctx.fillStyle = '#440154';
    ctx.fillRect(0, 0, w, h);

    const cellW = w / (HISTORY_LEN * 2);
    const cellH = h / N_BINS;

    // Use log scale (dB) for visibility, matching Python's visualize_curriculum_phases.py
    const { dbFrames, minDB, maxDB } = powerToDBNormalized(melHistory);

    for (let t = 0; t < dbFrames.length; t++) {
        for (let f = 0; f < N_BINS; f++) {
            const val = normalizeDB(dbFrames[t][f], minDB, maxDB);
            const [r, g, b] = getViridisColor(val);
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;

            const x = Math.floor(t * cellW);
            const y = Math.floor(h - (f + 1) * cellH);
            const nextX = Math.floor((t + 1) * cellW);
            const nextY = Math.floor(h - f * cellH);
            ctx.fillRect(x, y, nextX - x + 1, nextY - y + 1);
        }
    }
}

function drawProbs(canvas, history, labels, onlyTop = false, decodedEvents = [], totalFrames = 0) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);

    if (history.length === 0) return;

    const numClasses = history[0].length;
    const step = w / HISTORY_LEN;

    // Professional colors (Tableau10 style)
    const colors = ['#7f7f7f', '#d62728', '#2ca02c', '#1f77b4', '#bcbd22', '#e377c2', '#17becf', '#8c564b'];

    for (let c = 0; c < numClasses; c++) {
        if (onlyTop && c === 0) continue; // Skip blank for CTC if requested

        ctx.beginPath();
        ctx.strokeStyle = colors[c % colors.length];
        ctx.lineWidth = 1;

        for (let t = 0; t < history.length; t++) {
            const prob = history[t][c];
            const x = t * step;
            const y = h - (prob * h);
            if (t === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Draw Labels for Signal Canvas
    if (labels.length === 4) {
        ctx.font = '10px monospace';
        labels.forEach((l, i) => {
            ctx.fillStyle = colors[i % colors.length];
            ctx.fillText(l, 5, 12 + i * 12);
        });
    }

    // Draw Decoded Characters on CTC Canvas
    // Convert input frame units to output frame units (subsampling rate = 2)
    if (onlyTop && decodedEvents.length > 0) {
        ctx.font = 'bold 16px monospace';
        ctx.fillStyle = '#000';
        const outputFrames = Math.floor(totalFrames / 2);
        const currentBasePos = Math.max(0, outputFrames - HISTORY_LEN);
        decodedEvents.forEach(ev => {
            const evPosOutput = Math.floor(ev.pos / 2);
            const x = (evPosOutput - currentBasePos) * step;
            if (x > 0 && x < w) {
                const isSpace = ev.char === ' ';
                const displayChar = isSpace ? '␣' : ev.char;
                ctx.fillText(displayChar, x, 20);
                ctx.beginPath();
                // Use a more visible color and thicker line for decoded positions.
                // Highlight spaces with a different color (blue-ish).
                ctx.strokeStyle = isSpace ? 'rgba(0, 100, 255, 0.5)' : 'rgba(0, 0, 0, 0.4)';
                ctx.lineWidth = isSpace ? 2 : 1;
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
        });
    }
}

function drawBoundary(boundHistory = []) {
    const ctx = boundCanvas.getContext('2d');
    const w = boundCanvas.width;
    const h = boundCanvas.height;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);

    const step = w / HISTORY_LEN;
    ctx.beginPath();
    ctx.strokeStyle = '#1f77b4';
    ctx.lineWidth = 2;

    for (let t = 0; t < boundHistory.length; t++) {
        const prob = boundHistory[t];
        const x = t * step;
        const y = h - (prob * h);
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// --- Morse Logic ---
async function playMorse(text, signal) {
    const wpm = parseInt(wpmSlider.value);
    const jitter = parseFloat(jitterSlider.value);
    const dit = 1.2 / wpm;
    const attackRelease = 0.005; // 5ms

    const sleep = (ms) => new Promise((resolve, reject) => {
        const timeout = setTimeout(resolve, ms);
        signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new Error('aborted'));
        }, { once: true });
    });

    const getLen = (base) => base * dit * (1 + (Math.random() * 2 - 1) * jitter);

    try {
        // Initial delay to let the user prepare and buffers stabilize
        let schedTime = audioContext.currentTime + 0.1;

        const chars = text.toUpperCase().split("");
        for (let i = 0; i < chars.length; i++) {
            if (signal.aborted) return;
            const char = chars[i];
            
            if (char === ' ') {
                // Word space is 7 dits.
                // Since the previous character already added 3 dits (char space),
                // we add 4 more dits here.
                schedTime += getLen(4);
                continue;
            }

            const code = MORSE_DICT[char];
            if (!code) continue;

            for (let j = 0; j < code.length; j++) {
                const symbol = code[j];
                const duration = getLen(symbol === '.' ? 1 : 3);
                
                // Schedule ON
                gainNode.gain.setTargetAtTime(1.0, schedTime, attackRelease);
                schedTime += duration;
                
                // Schedule OFF
                gainNode.gain.setTargetAtTime(0.0, schedTime, attackRelease);
                
                // Inter-element space (1 dit)
                if (j < code.length - 1) {
                    schedTime += getLen(1);
                }
            }

            // Character space is 3 dits.
            // Since the last element already added 0 dits (just OFF),
            // and the next element or word space will start from schedTime,
            // we add 3 dits here.
            // BUT wait: the standard says 3 dits BETWEEN characters.
            // If the NEXT character is a space, we handle it in the ' ' block.
            // If the NEXT character is another letter, we need 3 dits.
            // Character space is 3 dits.
            // Add this after every character, regardless of what follows.
            if (chars[i + 1]) {
                schedTime += getLen(3);
            }
        }

        // Wait until the scheduled sequence finishes
        const totalDurationMs = (schedTime - audioContext.currentTime) * 1000;
        if (totalDurationMs > 0) {
            await sleep(totalDurationMs + 1000);
        }
        if (isRunning) stopBtn.onclick();
    } catch (e) {
        if (e.message !== 'aborted') throw e;
    }
}

function createWhiteNoise(durationSeconds, snr_2500) {
    const bufferSize = audioContext.sampleRate * durationSeconds;
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);

    // Match data_gen.js / data_gen.py:
    // SNR_2500 is defined relative to 2500Hz bandwidth.
    // C/N0 = SNR_2500 + 10 * log10(2500)
    const cn0 = snr_2500 + 10 * Math.log10(2500);

    // For a sine wave with amplitude 1.0, the power (C) is 0.5.
    const sigMarkWatts = 0.5;
    // N0 = C / 10^(cn0/10)
    const n0 = sigMarkWatts / Math.pow(10, cn0 / 10);
    // Total noise power in Nyquist bandwidth (Fs/2)
    const noiseAvgWatts = n0 * (audioContext.sampleRate / 2);
    const sigma = Math.sqrt(noiseAvgWatts);

    // Box-Muller transform for Gaussian noise
    for (let i = 0; i < bufferSize; i += 2) {
        const u1 = Math.random();
        const u2 = Math.random();
        const mag = sigma * Math.sqrt(-2.0 * Math.log(u1 || 1e-12)); // Avoid log(0)
        data[i] = mag * Math.cos(2.0 * Math.PI * u2);
        if (i + 1 < bufferSize) {
            data[i + 1] = mag * Math.sin(2.0 * Math.PI * u2);
        }
    }

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.loop = true;
    return source;
}

// --- Lifecycle ---
async function ensureAudioContext() {
    if (!audioContext) {
        // Note: Browsers may not respect the requested sampleRate
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        await audioContext.audioWorklet.addModule('audio-processor.js');
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    console.log(`AudioContext sampleRate: ${audioContext.sampleRate}Hz (Target: ${SAMPLE_RATE}Hz)`);
}

startBtn.onclick = async () => {
    if (isRunning) return;
    
    status.textContent = "初期化中...";
    await ensureAudioContext();

    resetBtn.onclick();
    
    isRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;

    // --- Receiver Pipeline ---
    morseNode = new AudioWorkletNode(audioContext, 'morse-processor', {
        processorOptions: {
            hopLength: HOP_LENGTH
        }
    });
    morseNode.connect(audioContext.destination);
    
    morseNode.port.onmessage = (e) => {
        if (!isRunning || !streamInference) return;
        if (e.data.type === 'audio_chunk') {
            audioBuffer.set(audioBuffer.subarray(HOP_LENGTH));
            audioBuffer.set(e.data.chunk, N_FFT - HOP_LENGTH);
            // Use computeSpecFrame from inference.js with fixed 16kHz (input is resampled)
            const mel = computeSpecFrame(audioBuffer, 16000);

            // Store for mel visualization
            melHistory.push(mel);
            if (melHistory.length > HISTORY_LEN * 2) melHistory.shift();

            // StreamInference handles buffering and inference internally
            // Chunk size validation (must be multiple of 4) is enforced by StreamInference
            streamInference.pushFrame(mel);
        }
    };

    // --- Sender Pipeline ---
    const freq = parseInt(freqSlider.value);

    // 1. Oscillator (Sine wave)
    oscillator = audioContext.createOscillator();
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(freq, audioContext.currentTime);
    
    // 2. Gain (Envelope)
    gainNode = audioContext.createGain();
    gainNode.gain.setValueAtTime(0, audioContext.currentTime);

    // 2.5 Master Gain (Volume Control)
    masterGainNode = audioContext.createGain();
    masterGainNode.gain.setValueAtTime(parseFloat(volumeSlider.value), audioContext.currentTime);
    
    // 3. Noise (White noise)
    noiseNode = createWhiteNoise(10, parseInt(snrSlider.value)); // snrSlider.value is SNR_2500
    
    // 4. Bandpass Filter (Radio-like CW filter, 500Hz bandwidth for inference)
    filterNode = audioContext.createBiquadFilter();
    filterNode.type = 'bandpass';
    filterNode.frequency.setValueAtTime(freq, audioContext.currentTime);
    // Q factor for ~500Hz bandwidth at 700Hz: Q = center_freq / bandwidth
    filterNode.Q.setValueAtTime(freq / 500, audioContext.currentTime);

    // 4.5 Output Filter (300Hz bandwidth for human hearing)
    outputFilterNode = audioContext.createBiquadFilter();
    outputFilterNode.type = 'bandpass';
    outputFilterNode.frequency.setValueAtTime(freq, audioContext.currentTime);
    // Q factor for ~300Hz bandwidth at 700Hz: Q = center_freq / bandwidth
    outputFilterNode.Q.setValueAtTime(freq / 300, audioContext.currentTime);

    // Connect Sender: Osc -> Gain -> Filter
    oscillator.connect(gainNode);
    gainNode.connect(filterNode);
    // Connect Noise: Noise -> Filter
    noiseNode.connect(filterNode);

    // Filter output goes to Master Gain
    filterNode.connect(masterGainNode);

    // Master Gain output goes to Receiver (for inference) and Output Filter (for hearing)
    masterGainNode.connect(morseNode);
    masterGainNode.connect(outputFilterNode);
    // Output Filter goes to audio destination (human hearing)
    outputFilterNode.connect(audioContext.destination);

    oscillator.start();
    noiseNode.start();

    status.textContent = "実行中";
    morseAbortController = new AbortController();
    playMorse(inputText.value, morseAbortController.signal);
};

stopBtn.onclick = () => {
    if (!isRunning) return;
    isRunning = false;
    
    if (morseAbortController) {
        morseAbortController.abort();
        morseAbortController = null;
    }
    
    if (oscillator) {
        try { oscillator.stop(); oscillator.disconnect(); } catch(e) {}
        oscillator = null;
    }
    if (gainNode) {
        gainNode.disconnect();
        gainNode = null;
    }
    if (masterGainNode) {
        masterGainNode.disconnect();
        masterGainNode = null;
    }
    if (noiseNode) {
        try { noiseNode.stop(); noiseNode.disconnect(); } catch(e) {}
        noiseNode = null;
    }
    if (filterNode) {
        filterNode.disconnect();
        filterNode = null;
    }
    if (outputFilterNode) {
        outputFilterNode.disconnect();
        outputFilterNode = null;
    }
    if (morseNode) {
        morseNode.disconnect();
        morseNode = null;
    }
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = "停止中";
};

resetBtn.onclick = () => {
    output.textContent = "";
    melHistory = [];
    skipCount = 0;

    // Reset StreamInference (handles state, decoder, history internally)
    if (streamInference) {
        streamInference.reset();
    }

    drawVisualizations();
    updateDebugInfo();
};

modelSelect.onchange = () => {
    if (isRunning) {
        stopBtn.onclick();
    }
    initONNX();
};

initONNX();