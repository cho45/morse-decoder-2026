/**
 * CW Decoder - Main Demo Script
 */
import {
    N_FFT, HOP_LENGTH, N_BINS, NUM_LAYERS, D_MODEL, N_HEAD, D_K, KERNEL_SIZE,
    CHARS, ID_TO_CHAR,
    computeSpecFrame, softmax, sigmoid, initStates, runChunkInference, ChunkedDecoder
} from './inference.js';
import { MORSE_DICT } from './data_gen.js';

// --- Constants & Config ---
const SAMPLE_RATE = 16000; // Model expected sample rate
const INPUT_LEN = 400; // Original window size (25ms @ 16kHz)
const INFERENCE_CHUNK_SIZE = 12; // 120ms。推論を実行するチャンクのフレーム数。
// 【重要】INFERENCE_CHUNK_SIZE は必ず 4 の倍数（SUBSAMPLING_RATE=2 の二乗）である必要があります。
// 4 の倍数でないチャンク（10等）を入力すると、runChunkInference がエラーを投げます。
// これは ONNX Runtime の制限によるものです：state updates が output より前に実行されるため、
// パディングを行うとモデルの内部キャッシュ（ConvSubsampling, ConformerConvModule, Attention）が
// 破損してしまいます。

// --- State Variables ---
let audioContext = null;
let morseNode = null;
let oscillator = null;
let gainNode = null;
let masterGainNode = null;
let noiseNode = null;
let filterNode = null;
let session = null;
let isRunning = false;
let isInferenceRunning = false; // 推論の重なりを防止
let currentStates = null;
let decoder = null; // ChunkedDecoder instance for streaming CTC decoding
let audioBuffer = new Float32Array(N_FFT); // Sliding window for FFT
let melBuffer = []; // Buffer for frames
let decodedEvents = []; // {char: string, pos: number} for visualization
let morseAbortController = null;
let totalFrames = 0;
let skipCount = 0;

// Visualization Buffers (History)
const HISTORY_LEN = 200; // Number of frames to show
let melHistory = [];
let ctcHistory = [];
let sigHistory = [];
let boundHistory = [];

// Canvas Contexts
const melCanvas = document.getElementById('melCanvas');
const ctcCanvas = document.getElementById('ctcCanvas');
const sigCanvas = document.getElementById('sigCanvas');
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
        
        // URLハッシュでプロバイダーを切り替え (#webgpu または #wasm)
        const useWebGPU = window.location.hash === '#webgpu';
        const providers = useWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
        
        // 既存のセッションがあれば（もし可能なら）破棄
        if (session) {
            // ort-web では明示的なセッション破棄メソッドはないが、
            // 変数を上書きすることでGC対象にする
            session = null;
        }

        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: providers
        });
        
        // 実際に使用されているプロバイダー名を取得
        let provider = 'unknown';
        if (session.executionProviders && session.executionProviders.length > 0) {
            provider = session.executionProviders[0];
        } else if (session.handler) {
            // handler の型名から判定を試みる (OnnxruntimeWebBackendWasm, OnnxruntimeWebBackendWebgpu など)
            const handlerName = session.handler.constructor.name.toLowerCase();
            if (handlerName.includes('webgpu')) provider = 'webgpu';
            else if (handlerName.includes('wasm')) provider = 'wasm';
            else provider = session.handler.epName || handlerName;
        }
        status.textContent = `準備完了 (${provider})`;
        debugInfo.innerHTML = `モデル: 読み込み完了 (${provider})<br>モード切替: URLに #webgpu または #wasm を追加<br>スキップ数: 0`;
    } catch (e) {
        status.textContent = "モデル読み込み失敗: " + e;
        console.error(e);
    }
}

function initStatesWrapper() {
    return initStates(ort);
}

async function runInference(melFrame, tLen = 1) {
    if (!session || !currentStates) return;
    if (isInferenceRunning) {
        // 推論が追いついていない場合はスキップ
        skipCount++;
        console.warn(`Inference skipped (busy) - Total skips: ${skipCount}`);
        updateDebugInfo();
        return;
    }
    isInferenceRunning = true;

    const start = performance.now();
    try {
        // Use the centralized runChunkInference to handle PCEN and states
        const result = await runChunkInference(session, melFrame, currentStates, ort);
        
        const oldStateValues = Object.values(currentStates);
        currentStates = result.nextStates;

        // WebGPU memory cleanup
        const nextStateValues = Object.values(currentStates);
        oldStateValues.forEach(t => {
            if (t && t.dispose && t.dims.length > 0 && t.dims.every(d => d > 0)) {
                if (!nextStateValues.includes(t)) {
                    t.dispose();
                }
            }
        });

    // Data for visualization and decoding
    const numOutFrames = result.logits.length / result.numClasses;

    for (let t = 0; t < numOutFrames; t++) {
        const ctcLogits = result.logits.slice(t * result.numClasses, (t + 1) * result.numClasses);
        const sigLogits = result.signalLogits.slice(t * 4, (t + 1) * 4);
        const boundLogit = result.boundaryLogits[t];

        const ctcProbs = softmax(Array.from(ctcLogits));
        const sigProbs = softmax(Array.from(sigLogits));
        const boundProb = sigmoid(boundLogit);

        // Visualization History
        ctcHistory.push(ctcProbs);
        sigHistory.push(sigProbs);
        boundHistory.push(boundProb);
        totalFrames++;

        if (ctcHistory.length > HISTORY_LEN) ctcHistory.shift();
        if (sigHistory.length > HISTORY_LEN) sigHistory.shift();
        if (boundHistory.length > HISTORY_LEN) boundHistory.shift();

        // CTC Decoding with ChunkedDecoder
        const decodeResult = decoder.decodeFrame(ctcLogits, sigLogits, boundProb);

        // Update UI and events when new character is emitted
        if (decodeResult.newChar) {
            // If space was inserted before this character, add space event
            if (decodeResult.spaceInserted) {
                decodedEvents.push({char: " ", pos: totalFrames - 1});
            }
            // Add character event
            decodedEvents.push({char: decodeResult.newChar, pos: totalFrames - 1});
            output.textContent = decodeResult.text;
        }
    }

    // Mel History (always 2 frames for the chunk)
    for (let t = 0; t < tLen; t++) {
        melHistory.push(melFrame.slice(t * N_BINS, (t + 1) * N_BINS));
        if (melHistory.length > HISTORY_LEN * 2) melHistory.shift();
    }
    
    // Cleanup decoded events that are out of history
    while (decodedEvents.length > 0 && decodedEvents[0].pos < totalFrames - HISTORY_LEN) {
        decodedEvents.shift();
    }

    drawVisualizations();
    updateDebugInfo(start);

    } catch (e) {
        console.error("Inference error:", e);
    } finally {
        isInferenceRunning = false;
    }
}

function updateDebugInfo(startTime = null) {
    if (!session || !currentStates) return;
    
    let provider = 'unknown';
    if (session.executionProviders && session.executionProviders.length > 0) {
        provider = session.executionProviders[0];
    } else if (session.handler) {
        const handlerName = session.handler.constructor.name.toLowerCase();
        if (handlerName.includes('webgpu')) provider = 'webgpu';
        else if (handlerName.includes('wasm')) provider = 'wasm';
        else provider = session.handler.epName || 'unknown';
    }

    const offset = currentStates.offset_0.data[0];
    const offsetNum = typeof offset === 'bigint' ? Number(offset) : offset;
    
    let latencyInfo = "";
    if (startTime) {
        const end = performance.now();
        latencyInfo = `<br>レイテンシ: ${(end - startTime).toFixed(1)}ms`;
    }

    debugInfo.innerHTML = `モデル: ${isRunning ? '動作中' : '読み込み完了'} (${provider})<br>` +
                         `処理フレーム数: ${Math.floor(offsetNum)}<br>` +
                         `スキップ数: ${skipCount}${latencyInfo}`;
}


function drawVisualizations() {
    drawMel();
    drawProbs(ctcCanvas, ctcHistory, ["blank", ...CHARS], true);
    drawProbs(sigCanvas, sigHistory, ["Space", "Dit", "Dah", "Inter-Word"]);
    drawBoundary();
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

    // Accurate Viridis colormap points
    const viridis = [
        [68, 1, 84],   // 0.0
        [72, 40, 120], // 0.125
        [62, 74, 137], // 0.25
        [49, 104, 142], // 0.375
        [38, 130, 142], // 0.5
        [31, 158, 137], // 0.625
        [53, 183, 121], // 0.75
        [109, 205, 89], // 0.875
        [253, 231, 37]  // 1.0
    ];

    const getColor = (v) => {
        v = Math.max(0, Math.min(1, v));
        const pos = v * (viridis.length - 1);
        const idx = Math.floor(pos);
        const frac = pos - idx;
        if (idx >= viridis.length - 1) return viridis[viridis.length - 1];
        
        const c1 = viridis[idx];
        const c2 = viridis[idx + 1];
        return [
            Math.floor(c1[0] + (c2[0] - c1[0]) * frac),
            Math.floor(c1[1] + (c2[1] - c1[1]) * frac),
            Math.floor(c1[2] + (c2[2] - c1[2]) * frac)
        ];
    };

    // Use log scale (dB) for visibility, matching Python's visualize_curriculum_phases.py
    // Calculate adaptive range in dB space
    let minDB = Infinity;
    let maxDB = -Infinity;
    const dbHistory = melHistory.map(frame => {
        return Array.from(frame).map(p => 10 * Math.log10(p + 1e-9));
    });

    for (let t = 0; t < dbHistory.length; t++) {
        for (let f = 0; f < N_BINS; f++) {
            const db = dbHistory[t][f];
            if (db < minDB) minDB = db;
            if (db > maxDB) maxDB = db;
        }
    }
    // Ensure a minimum range to avoid division by zero
    if (maxDB <= minDB) maxDB = minDB + 1;

    for (let t = 0; t < dbHistory.length; t++) {
        for (let f = 0; f < N_BINS; f++) {
            const db = dbHistory[t][f];
            const val = (db - minDB) / (maxDB - minDB);
            const [r, g, b] = getColor(val);
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            
            const x = Math.floor(t * cellW);
            const y = Math.floor(h - (f + 1) * cellH);
            const nextX = Math.floor((t + 1) * cellW);
            const nextY = Math.floor(h - f * cellH);
            ctx.fillRect(x, y, nextX - x + 1, nextY - y + 1);
        }
    }
}

function drawProbs(canvas, history, labels, onlyTop = false) {
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
    if (onlyTop) {
        ctx.font = 'bold 16px monospace';
        ctx.fillStyle = '#000';
        const currentBasePos = Math.max(0, totalFrames - HISTORY_LEN);
        decodedEvents.forEach(ev => {
            const x = (ev.pos - currentBasePos) * step;
            if (x > 0 && x < w) {
                ctx.fillText(ev.char, x, 20);
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
        });
    }
}

function drawBoundary() {
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
                // we only need to add 4 more dits here.
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
            const nextChar = chars[i + 1];
            if (nextChar && nextChar !== ' ') {
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

function createWhiteNoise(durationSeconds, snr) {
    const bufferSize = audioContext.sampleRate * durationSeconds;
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);

    // Match data_gen.py: SNR is based on the average power of the signal during the MARK (ON) state.
    // For a sine wave with amplitude 1.0, the power is 0.5.
    const sigMarkWatts = 0.5;
    const noiseAvgWatts = sigMarkWatts / Math.pow(10, snr / 10);
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
        if (!isRunning) return;
        if (e.data.type === 'audio_chunk') {
            audioBuffer.set(audioBuffer.subarray(HOP_LENGTH));
            audioBuffer.set(e.data.chunk, N_FFT - HOP_LENGTH);
            // Use computeSpecFrame from inference.js with fixed 16kHz (input is resampled)
            const mel = computeSpecFrame(audioBuffer, 16000);
            melBuffer.push(mel);
            // チャンクサイズごとに推論を実行。
            // 【重要】INFERENCE_CHUNK_SIZE は必ず 4 の倍数（SUBSAMPLING_RATE=2 の二乗）である必要があります。
            // 4 の倍数でないチャンク（10等）を入力すると、runChunkInference がエラーを投げます。
            // これは ONNX Runtime の制限によるものです：state updates が output より前に実行されるため、
            // パディングを行うとモデルの内部キャッシュ（ConvSubsamplingやConformerConvModuleの状態）が
            // 破損してしまいます。
            if (melBuffer.length >= INFERENCE_CHUNK_SIZE) {
                const combinedMel = new Float32Array(N_BINS * INFERENCE_CHUNK_SIZE);
                for (let i = 0; i < INFERENCE_CHUNK_SIZE; i++) {
                    combinedMel.set(melBuffer[i], i * N_BINS);
                }
                runInference(combinedMel, INFERENCE_CHUNK_SIZE);
                melBuffer = [];
            }
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
    noiseNode = createWhiteNoise(10, parseInt(snrSlider.value));
    
    // 4. Bandpass Filter (Radio-like CW filter)
    filterNode = audioContext.createBiquadFilter();
    filterNode.type = 'bandpass';
    filterNode.frequency.setValueAtTime(freq, audioContext.currentTime);
    // Q factor for ~500Hz bandwidth at 700Hz: Q = center_freq / bandwidth
    filterNode.Q.setValueAtTime(freq / 500, audioContext.currentTime);

    // Connect Sender: Osc -> Gain -> Filter
    oscillator.connect(gainNode);
    gainNode.connect(filterNode);
    // Connect Noise: Noise -> Filter
    noiseNode.connect(filterNode);

    // Filter output goes to Master Gain
    filterNode.connect(masterGainNode);

    // Master Gain output goes to both Receiver (for inference) and Destination (for hearing)
    masterGainNode.connect(morseNode);
    masterGainNode.connect(audioContext.destination);

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
    currentStates = initStatesWrapper();
    // Initialize decoder with the correct number of classes
    decoder = new ChunkedDecoder(CHARS.length + 1); // +1 for blank
    melBuffer = [];
    melHistory = [];
    ctcHistory = [];
    sigHistory = [];
    boundHistory = [];
    decodedEvents = [];
    totalFrames = 0;
    skipCount = 0;
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