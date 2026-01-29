/**
 * CW Decoder - Microphone & Wide-band Demo Script
 */
import { DSP } from './dsp.js';
import { NoiseNode } from './noise-node.js';
import { StreamInference } from './stream-inference.js';
import { MORSE_DICT } from './data_gen.js';
import { getViridisColor, powerToDBNormalized, normalizeDB } from './visualization.js';

// --- Constants & Config ---
const WINDOW_MS = 32;
const HOP_MS = 10;
const MAX_FREQ = 4000;
const PEAK_LOCK_MS = 3000;
const HISTORY_LEN = 800; // History length for StreamInference

// --- State Variables ---
let audioContext = null;
let session = null;
let streamInference = null; // StreamInference instance
let isRunning = false;
let stream = null;
let demoNodes = [];
let masterGainNode = null;
let userFilterNode = null;

// FFT & Processing Parameters
const TARGET_SAMPLE_RATE = 16000;
let sampleRate = 0; // Actual AudioContext sample rate
let windowSize = 0;
let hopLength = 0;
let nFft = 0;

// Peak Tracking State
let targetFreq = 800;
let trackedFreq = 800;
let lastTrackedFreq = 800;
let peakLockTimer = 0;

// Buffers
const historyLen = 400;
let waterfallBuffer = [];
let audioBuf = null;
let inputSpecBuffer = [];
let inputSpecHistory = []; // Buffer for 14 bins history (for visualization only)
let capturedSpecData = []; // Buffer for exporting inference data

// UI Elements
const waterfallCanvas = document.getElementById('waterfallCanvas');
const waterfallCtx = waterfallCanvas.getContext('2d', { alpha: false });
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas.getContext('2d');
const inputCanvas = document.getElementById('inputCanvas');
const inputCtx = inputCanvas.getContext('2d', { alpha: false });
const inputOverlayCanvas = document.getElementById('inputOverlayCanvas');
const inputOverlayCtx = inputOverlayCanvas.getContext('2d');
const micBtn = document.getElementById('micBtn');
const demoBtn = document.getElementById('demoBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const debugInfo = document.getElementById('debugInfo');
const volumeSlider = document.getElementById('volume');
const volumeValue = document.getElementById('volumeValue');
const autoTrackCheck = document.getElementById('autoTrack');
const exportBtn = document.getElementById('exportBtn');

// Viridis colormap is imported from visualization.js

// --- Station Class for Demo Mode ---

class Station {
    constructor(ctx, freq, wpm, jitter, volume, destination) {
        this.ctx = ctx;
        this.freq = freq;
        this.wpm = wpm;
        this.jitter = jitter;
        this.volume = volume;
        this.osc = ctx.createOscillator();
        this.gain = ctx.createGain();
        this.osc.type = 'sine';
        this.osc.frequency.value = freq;
        this.gain.gain.value = 0;
        this.osc.connect(this.gain);
        this.gain.connect(destination);
        this.osc.start();
        this.active = true;
    }

    async play(text) {
        await new Promise(r => setTimeout(r, Math.random() * 5000));
        const attackRelease = 0.005;
        const ditBase = 1.2 / this.wpm;
        const getLen = (units) => units * ditBase * (1 + (Math.random() * 2 - 1) * this.jitter);

        while (this.active && isRunning) {
            let schedTime = this.ctx.currentTime + 0.1;
            for (const char of text.toUpperCase()) {
                if (!this.active || !isRunning) break;
                if (char === ' ') { schedTime += getLen(4); continue; }
                const code = MORSE_DICT[char];
                if (!code) continue;
                for (const symbol of code) {
                    const duration = getLen(symbol === '.' ? 1 : 3);
                    this.gain.gain.setTargetAtTime(this.volume, schedTime, attackRelease);
                    schedTime += duration;
                    this.gain.gain.setTargetAtTime(0, schedTime, attackRelease);
                    schedTime += getLen(1);
                }
                schedTime += getLen(2); // char space
            }
            await new Promise(r => setTimeout(r, Math.max(0, (schedTime - this.ctx.currentTime) * 1000) + 2000));
        }
    }

    stop() {
        this.active = false;
        try { this.osc.stop(); this.osc.disconnect(); this.gain.disconnect(); } catch (e) {}
    }
}

// --- ONNX Inference ---

async function initONNX() {
    try {
        const modelPath = document.getElementById('modelSelect').value;
        status.textContent = `モデル読み込み中...`;

        // Dispose existing StreamInference
        if (streamInference) {
            streamInference.dispose();
            streamInference = null;
        }

        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['wasm']
        });

        // Create StreamInference instance
        streamInference = new StreamInference(session, ort, {
            chunkSize: 12,
            useWebGPU: false,
            historyLength: HISTORY_LEN
        });

        // Set up event listeners
        streamInference.addEventListener('result', (e) => {
            document.getElementById('output').textContent = e.detail.text;
        });

        status.textContent = `準備完了`;
    } catch (e) {
        status.textContent = "モデル読み込み失敗: " + e;
        console.error(e);
    }
}

// StreamInference handles all inference state management, decoding, and history internally.
// We only need to push frames via streamInference.pushFrame() and listen for events.

// --- Core Logic ---

function updateUserFilter() {
    if (userFilterNode && Math.abs(trackedFreq - lastTrackedFreq) > 10) {
        userFilterNode.frequency.setTargetAtTime(trackedFreq, audioContext.currentTime, 0.1);
        lastTrackedFreq = trackedFreq;
    }
}

function peakDetect(magnitudes) {
    const numBins = magnitudes.length;
    let peaks = [];
    let sumP = 0;
    for (let j = 0; j < numBins; j++) sumP += magnitudes[j];
    const avgP = sumP / numBins;
    const threshold = avgP * 5 + 0.0001;

    for (let j = 1; j < numBins - 1; j++) {
        if (magnitudes[j] > threshold && magnitudes[j] > magnitudes[j - 1] && magnitudes[j] > magnitudes[j + 1]) {
            peaks.push({ p: magnitudes[j], k: j, f: j * TARGET_SAMPLE_RATE / nFft });
        }
    }
    peaks.sort((a, b) => b.p - a.p);

    if (autoTrackCheck.checked) {
        const now = Date.now();
        const nearbyPeak = peaks.find(p => Math.abs(p.f - trackedFreq) < 50);
        if (nearbyPeak) {
            trackedFreq = trackedFreq * 0.9 + nearbyPeak.f * 0.1;
            peakLockTimer = now + PEAK_LOCK_MS;
        } else if (now > peakLockTimer && peaks.length > 0) {
            trackedFreq = peaks[0].f;
            peakLockTimer = now + PEAK_LOCK_MS;
        }
    } else {
        trackedFreq = targetFreq;
    }
    updateUserFilter();
}

function processAudioChunk(chunk) {
    audioBuf.set(audioBuf.subarray(chunk.length));
    audioBuf.set(chunk, nFft - chunk.length);

    const real = new Float32Array(nFft);
    real.set(audioBuf.subarray(nFft - windowSize));
    const imag = new Float32Array(nFft);
    for (let j = 0; j < windowSize; j++) {
        real[j] *= 0.5 * (1 - Math.cos(2 * Math.PI * j / windowSize));
    }
    DSP.fft(real, imag);

    // Note: magnitudes for peak detection are still based on the resampled 16kHz stream
    // but nFft/windowSize are calculated based on 16kHz.
    const numBins = Math.floor(MAX_FREQ * nFft / TARGET_SAMPLE_RATE);
    const magnitudes = new Float32Array(numBins);
    for (let j = 0; j < numBins; j++) magnitudes[j] = real[j] * real[j] + imag[j] * imag[j];

    // 1. Peak Detection
    peakDetect(magnitudes);

    // 2. Inference Sampling - Extract 14 bins centered on tracked frequency
    const binBW = TARGET_SAMPLE_RATE / nFft;
    const W = 14 * binBW;
    const fStart = trackedFreq - W/2 + binBW/2;
    const specFrame = new Float32Array(14);
    for (let i = 0; i < 14; i++) {
        const f = fStart + i * binBW;
        const k = f * nFft / TARGET_SAMPLE_RATE;
        const kIdx = Math.floor(k);
        const kFrac = k - kIdx;
        if (kIdx >= 0 && kIdx < nFft - 1) {
            const p1 = real[kIdx]*real[kIdx] + imag[kIdx]*imag[kIdx];
            const p2 = real[kIdx+1]*real[kIdx+1] + imag[kIdx+1]*imag[kIdx+1];
            const p = p1 * (1 - kFrac) + p2 * kFrac;

            specFrame[i] = p;
        }
    }

    // Export data capture
    capturedSpecData.push({
        frame: Array.from(specFrame),
        freq: trackedFreq
    });
    if (capturedSpecData.length === 1) exportBtn.disabled = false;

    // StreamInference handles buffering and inference internally
    // Chunk size validation (must be multiple of 4) is enforced by StreamInference
    if (streamInference) {
        streamInference.pushFrame(specFrame);
    }

    // Capture history for input visualization
    const frameCopy = new Float32Array(specFrame);
    inputSpecHistory.push(frameCopy);
    inputSpecBuffer.push(frameCopy);
    if (inputSpecHistory.length > 800) inputSpecHistory.shift();

    // 3. Waterfall update
    const displayMagnitude = magnitudes.map(p => Math.max(0, Math.log1p(p * 5000) / 12));
    waterfallBuffer.push(displayMagnitude);
    if (waterfallBuffer.length > historyLen) waterfallBuffer.shift();
}

// --- Lifecycle & UI ---

async function initAudio() {
    if (!session) await initONNX();
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        sampleRate = audioContext.sampleRate;
        await NoiseNode.addModule(audioContext);
        
        // windowSize/hopLength are now based on 16kHz
        windowSize = Math.floor(TARGET_SAMPLE_RATE * (WINDOW_MS / 1000));
        hopLength = Math.floor(TARGET_SAMPLE_RATE * (HOP_MS / 1000));
        nFft = Math.pow(2, Math.ceil(Math.log2(windowSize)));
        audioBuf = new Float32Array(nFft);
        debugInfo.innerHTML = `入力レート: ${sampleRate} Hz -> 処理レート: ${TARGET_SAMPLE_RATE} Hz | FFTサイズ: ${nFft} | フレーム間隔: ${HOP_MS} ms`;
        await audioContext.audioWorklet.addModule('audio-processor.js');
    }
    if (audioContext.state === 'suspended') await audioContext.resume();
}

function stopAll() {
    isRunning = false;

    // Reset StreamInference (handles state, decoder, history internally)
    if (streamInference) {
        streamInference.reset();
    }

    // Clear local visualization buffers
    waterfallBuffer = [];
    inputSpecBuffer = [];
    inputSpecHistory = [];

    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    demoNodes.forEach(n => { if (n.stop) n.stop(); if (n.disconnect) n.disconnect(); });
    demoNodes = [];
    masterGainNode = null;
    if (userFilterNode) {
        userFilterNode.disconnect();
        userFilterNode = null;
    }
    micBtn.disabled = false; demoBtn.disabled = false; stopBtn.disabled = true;
    status.textContent = "停止中";
}

micBtn.onclick = async () => {
    capturedSpecData = [];
    exportBtn.disabled = true;
    await initAudio();
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        setupProcessing(source);

        // Create bandpass filter for user listening (analysis path remains unfiltered)
        userFilterNode = audioContext.createBiquadFilter();
        userFilterNode.type = 'bandpass';
        userFilterNode.frequency.value = trackedFreq;
        userFilterNode.Q.value = 10;

        const outputGain = audioContext.createGain();
        outputGain.gain.value = parseFloat(volumeSlider.value);
        masterGainNode = outputGain;

        source.connect(userFilterNode);
        userFilterNode.connect(outputGain);
        outputGain.connect(audioContext.destination);

        isRunning = true;
        micBtn.disabled = true; demoBtn.disabled = true; stopBtn.disabled = false;
        status.textContent = "マイク動作中";
    } catch (e) { status.textContent = "マイクアクセス失敗: " + e; }
};

demoBtn.onclick = async () => {
    capturedSpecData = [];
    exportBtn.disabled = true;
    await initAudio();
    isRunning = true;
    micBtn.disabled = true; demoBtn.disabled = true; stopBtn.disabled = false;
    status.textContent = "デモモード動作中";

    const analysisMix = audioContext.createGain();
    masterGainNode = audioContext.createGain();
    masterGainNode.gain.value = parseFloat(volumeSlider.value);

    // Create bandpass filter for user listening (centered on trackedFreq)
    userFilterNode = audioContext.createBiquadFilter();
    userFilterNode.type = 'bandpass';
    userFilterNode.frequency.value = trackedFreq;
    userFilterNode.Q.value = 10; // Narrow bandwidth for focusing on single CW signal

    masterGainNode.connect(userFilterNode);
    userFilterNode.connect(audioContext.destination);

    const noise = new NoiseNode(audioContext, { type: 'whitenoise' });
    const noiseGain = audioContext.createGain();
    noiseGain.gain.value = 0.01;
    noise.connect(noiseGain);
    noiseGain.connect(analysisMix);
    noiseGain.connect(masterGainNode);
    demoNodes.push(noise);

    const stations = [
        { freq: 650,  wpm: 18, jitter: 0.05, vol: 0.3, msg: "CQ CQ DE JA1ABC K" },
        { freq: 1200, wpm: 25, jitter: 0.1,  vol: 0.1, msg: "CQ CQ DE K1XYZ K" },
        { freq: 1800, wpm: 35, jitter: 0.02, vol: 0.5, msg: "CQ CQ DE G4ZOO K" },
        { freq: 2500, wpm: 20, jitter: 0.15, vol: 0.05, msg: "CQ CQ DE JH1UMV K" },
        { freq: 3200, wpm: 28, jitter: 0.05, vol: 0.2, msg: "CQ CQ DE DF7CB K" }
    ];

    stations.forEach(s => {
        const st = new Station(audioContext, s.freq, s.wpm, s.jitter, s.vol, analysisMix);
        st.gain.connect(masterGainNode);
        st.play(s.msg);
        demoNodes.push(st);
    });

    setupProcessing(analysisMix);
};

stopBtn.onclick = stopAll;

function setupProcessing(source) {
    const workletNode = new AudioWorkletNode(audioContext, 'morse-processor', {
        processorOptions: {
            sampleRate: audioContext.sampleRate,
            hopLength: hopLength
        }
    });
    workletNode.port.onmessage = (e) => {
        if (isRunning && e.data.type === 'audio_chunk') processAudioChunk(e.data.chunk);
    };
    source.connect(workletNode);
    demoNodes.push(workletNode);
}

// --- Visualization ---

function drawWaterfall() {
    if (!isRunning) {
        requestAnimationFrame(drawWaterfall);
        return;
    }
    if (waterfallBuffer.length > 0) {
        const w = waterfallCanvas.width;
        const h = waterfallCanvas.height;
        const numFramesToDraw = waterfallBuffer.length;
        if (numFramesToDraw > 0) {
            waterfallCtx.drawImage(waterfallCanvas, -numFramesToDraw, 0);
            for (let f = 0; f < numFramesToDraw; f++) {
                const frameData = waterfallBuffer[f];
                const numBins = frameData.length;
                const binH = h / numBins;
                const x = w - numFramesToDraw + f;
                for (let i = 0; i < numBins; i++) {
                    const [r, g, b] = getViridisColor(frameData[i]);
                    waterfallCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    waterfallCtx.fillRect(x, h - (i + 1) * binH, 1, binH + 1);
                }
            }
            drawOverlay(w, h);
            waterfallBuffer = [];
        }
    }
    drawInputSpec();
    requestAnimationFrame(drawWaterfall);
}

function drawInputSpec() {
    if (!isRunning || inputSpecHistory.length === 0) return;
    const w = inputCanvas.width;
    const h = inputCanvas.height;
    
    if (inputSpecBuffer.length > 0) {
        const numToDraw = inputSpecBuffer.length;
        // 1. Shift Background by exactly the number of new frames
        inputCtx.drawImage(inputCanvas, -numToDraw, 0);

        // 2. Use log scale (dB) for visibility, matching Python's visualize_curriculum_phases.py
        const historyToScan = inputSpecHistory.slice(-w);
        const { minDB, maxDB } = powerToDBNormalized(historyToScan);

        const binH = h / 14;
        for (let f = 0; f < numToDraw; f++) {
            const frame = inputSpecBuffer[f];
            const x = w - numToDraw + f;
            for (let i = 0; i < 14; i++) {
                const db = 10 * Math.log10(frame[i] + 1e-9);
                const normalizedVal = normalizeDB(db, minDB, maxDB);
                const [r, g, b] = getViridisColor(normalizedVal);
                inputCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                inputCtx.fillRect(x, h - (i + 1) * binH, 1, binH + 1);
            }
        }
        inputSpecBuffer = [];
    }

    // 3. Draw Overlay (Signal classification & Decoded chars)
    drawInputOverlay(w, h);
}

function drawInputOverlay(w, h) {
    inputOverlayCtx.clearRect(0, 0, w, h);

    if (!streamInference) return;

    // Get history from StreamInference
    const sigHistory = streamInference.getSignalHistory();
    const eventHistory = streamInference.getEvents();
    const totalFrames = streamInference.frameCount;

    if (sigHistory.length === 0) return;

    const barH = 12;
    const sigColors = ['rgba(0,0,0,0)', '#ff4d4d', '#4d79ff', '#ffcc00'];

    sigHistory.forEach(item => {
        const x = w - (totalFrames - item.pos);
        if (x < 0 || x >= w) return;

        let maxIdx = 0, maxP = -1;
        for (let s = 0; s < 4; s++) {
            if (item.probs[s] > maxP) { maxP = item.probs[s]; maxIdx = s; }
        }

        if (maxIdx > 0) {
            inputOverlayCtx.fillStyle = sigColors[maxIdx];
            // Each sig output represents 2 input frames
            inputOverlayCtx.fillRect(x - 1, h - barH, 2, barH);
        }
    });

    // Draw Decoded Characters
    inputOverlayCtx.fillStyle = '#fff';
    inputOverlayCtx.font = 'bold 16px Courier New';
    inputOverlayCtx.textAlign = 'center';

    eventHistory.forEach(ev => {
        const x = w - (totalFrames - ev.pos);
        if (x > 0 && x < w) {
            inputOverlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            inputOverlayCtx.beginPath();
            inputOverlayCtx.moveTo(x, 0);
            inputOverlayCtx.lineTo(x, h - barH);
            inputOverlayCtx.stroke();
            inputOverlayCtx.fillText(ev.char, x, 20);
        }
    });
}

function drawOverlay(w, h) {
    overlayCtx.clearRect(0, 0, w, h);
    // 14 bins * binBW Hz/bin
    const binBW = TARGET_SAMPLE_RATE / nFft;
    const BW = 14 * binBW;
    
    // 座標計算。MAX_FREQ に対する trackedFreq の比率で計算する。
    const yCenter = h - (trackedFreq / MAX_FREQ) * h;
    const yHalfWidth = (BW / 2 / MAX_FREQ) * h;

    overlayCtx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    overlayCtx.fillRect(0, yCenter - yHalfWidth, w, yHalfWidth * 2);
    overlayCtx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
    overlayCtx.lineWidth = 2;
    overlayCtx.beginPath();
    overlayCtx.moveTo(w - 100, yCenter); overlayCtx.lineTo(w, yCenter);
    overlayCtx.stroke();

    overlayCtx.fillStyle = '#fff';
    overlayCtx.font = 'bold 14px Arial';
    overlayCtx.textAlign = 'right';
    overlayCtx.shadowColor = 'black'; overlayCtx.shadowBlur = 4;
    overlayCtx.fillText(`${Math.round(trackedFreq)} Hz`, w - 10, yCenter - 8);
    overlayCtx.shadowBlur = 0;

    overlayCtx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    overlayCtx.font = '10px Arial';
    overlayCtx.textAlign = 'left';
    for (let f = 0; f <= MAX_FREQ; f += 500) {
        const y = h - (f / MAX_FREQ) * h;
        if (y < 0 || y > h) continue;
        overlayCtx.beginPath(); overlayCtx.moveTo(0, y); overlayCtx.lineTo(15, y);
        overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)'; overlayCtx.stroke();
        overlayCtx.fillText(`${f}Hz`, 18, y + 3);
    }
}

overlayCanvas.onclick = (e) => {
    if (!sampleRate) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    targetFreq = (rect.height - y) / rect.height * MAX_FREQ;
    trackedFreq = targetFreq;
    lastTrackedFreq = targetFreq;
    autoTrackCheck.checked = false;

    // Update filter immediately
    if (userFilterNode) {
        userFilterNode.frequency.value = trackedFreq;
    }

    // Reset decoder state and text when changing frequency
    document.getElementById('output').textContent = "";
    if (streamInference) {
        streamInference.reset();
    }

    status.textContent = `周波数固定: ${Math.round(targetFreq)} Hz`;
};

volumeSlider.oninput = () => {
    const vol = parseFloat(volumeSlider.value);
    volumeValue.textContent = vol.toFixed(2);
    if (masterGainNode) masterGainNode.gain.setTargetAtTime(vol, audioContext.currentTime, 0.01);
};

function exportData() {
    if (capturedSpecData.length === 0) {
        alert("エクスポートするデータがありません。");
        return;
    }

    const exportObj = {
        sample_rate: sampleRate,
        n_fft: nFft,
        hop_ms: HOP_MS,
        n_bins: 14,
        data: capturedSpecData
    };

    const blob = new Blob([JSON.stringify(exportObj)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cw-decoder-spec-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

exportBtn.onclick = exportData;

drawWaterfall();