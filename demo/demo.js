/**
 * CW Decoder - Main Demo Script
 */

// --- Constants & Config (Sync with config.py) ---
let SAMPLE_RATE = 16000; // Updated by AudioContext
const N_FFT = 512; // Power of 2 for FFT
const INPUT_LEN = 400; // Original window size (25ms @ 16kHz)
const HOP_LENGTH = 160;
const N_MELS = 16;
const MAX_CACHE_LEN = 1000;
const NUM_LAYERS = 4;
const D_MODEL = 256;
const N_HEAD = 4;
const D_K = D_MODEL / N_HEAD;
const KERNEL_SIZE = 31;

const MORSE_MAP = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', '/': '-..-.', '?': '..--..', '.': '.-.-.-',
    ',': '--..--', ' ': ' '
};

// CTC Vocab (Sync with config.py)
const STD_CHARS = ",./0123456789?ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const PROSIGNS = ["<BT>", "<AR>", "<SK>", "<KA>"];
const CHARS = [...STD_CHARS, ...PROSIGNS];
const ID_TO_CHAR = {};
CHARS.forEach((char, i) => { ID_TO_CHAR[i + 1] = char; });

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
let audioBuffer = new Float32Array(N_FFT); // Sliding window for FFT
let melFilters = null;
let melBuffer = []; // Buffer for 2 frames
let decodedText = "";
let lastCharId = 0;
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

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// --- UI Elements ---
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
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

// --- DSP Utils ---
function applyFFTAndMel(samples, filters) {
    const n = N_FFT;
    const real = new Float32Array(n);
    const imag = new Float32Array(n);

    // Apply Hann window and copy to real buffer (with zero padding to 512)
    for (let i = 0; i < INPUT_LEN; i++) {
        real[i] = samples[i] * (0.5 * (1 - Math.cos(2 * Math.PI * i / (INPUT_LEN - 1))));
    }

    DSP.fft(real, imag);

    const num_bins = Math.floor(n / 2) + 1;
    const power = new Float32Array(num_bins);
    for (let k = 0; k < num_bins; k++) {
        power[k] = real[k] * real[k] + imag[k] * imag[k];
    }

    const mels = new Float32Array(filters.length);
    for (let i = 0; i < filters.length; i++) {
        let sum = 0;
        for (let j = 0; j < num_bins; j++) {
            sum += power[j] * filters[i][j];
        }
        // Match train.py/evaluate_detailed.py scaling: log1p(mel * 100) / 5.0
        mels[i] = Math.log1p(sum * 100.0) / 5.0;
    }
    return mels;
}

function createMelFilters(n_mels, n_fft, sample_rate) {
    return DSP.createMelFilters(n_mels, n_fft, sample_rate);
}

// --- ONNX Inference ---
async function initONNX() {
    try {
        status.textContent = "モデル読み込み中...";
        
        // URLハッシュでプロバイダーを切り替え (#webgpu または #wasm)
        const useWebGPU = window.location.hash === '#webgpu';
        const providers = useWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
        
        session = await ort.InferenceSession.create('./cw_decoder.onnx', {
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

function initStates() {
    const states = {
        sub_cache: new ort.Tensor('float32', new Float32Array(1 * 1 * 2 * N_MELS), [1, 1, 2, N_MELS])
    };
    for (let i = 0; i < NUM_LAYERS; i++) {
        states[`attn_k_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, N_HEAD, 0, D_K]);
        states[`attn_v_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, N_HEAD, 0, D_K]);
        states[`offset_${i}`] = new ort.Tensor('int64', new BigInt64Array([0n]), []);
        states[`conv_cache_${i}`] = new ort.Tensor('float32', new Float32Array(1 * D_MODEL * (KERNEL_SIZE - 1)), [1, D_MODEL, KERNEL_SIZE - 1]);
    }
    return states;
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
    let x = null;
    try {
        x = new ort.Tensor('float32', melFrame, [1, tLen, N_MELS]);
        const inputs = { x, ...currentStates };

        const results = await session.run(inputs);
        
        // 新しい状態をセットし、古い状態をリストアップ
        const oldStateValues = Object.values(currentStates);
        
        const nextStates = {};
        nextStates.sub_cache = results.new_sub_cache;
        for (let i = 0; i < NUM_LAYERS; i++) {
            nextStates[`attn_k_${i}`] = results[`new_attn_k_${i}`];
            nextStates[`attn_v_${i}`] = results[`new_attn_v_${i}`];
            nextStates[`offset_${i}`] = results[`new_offset_${i}`];
            nextStates[`conv_cache_${i}`] = results[`new_conv_cache_${i}`];
        }
        currentStates = nextStates;

        // 古い状態テンソルを明示的に破棄 (WebGPU メモリ解放)
        // 初期状態の空テンソルや、新しい状態と同一のインスタンスは破棄しない
        const nextStateValues = Object.values(currentStates);
        oldStateValues.forEach(t => {
            if (t && t.dispose && t.dims.length > 0 && t.dims.every(d => d > 0)) {
                if (!nextStateValues.includes(t)) {
                    t.dispose();
                }
            }
        });

    // Data for visualization (only last frame of the chunk for simplicity in history)
    const numOutFrames = results.logits.dims[1];
    for (let t = 0; t < numOutFrames; t++) {
        const ctcLogits = results.logits.data.slice(t * (CHARS.length + 1), (t + 1) * (CHARS.length + 1));
        const sigLogits = results.signal_logits.data.slice(t * 6, (t + 1) * 6);
        const boundLogit = results.boundary_logits.data[t];

        const ctcProbs = softmax(Array.from(ctcLogits));
        const sigProbs = softmax(Array.from(sigLogits));
        const boundProb = sigmoid(boundLogit);

        ctcHistory.push(ctcProbs);
        sigHistory.push(sigProbs);
        boundHistory.push(boundProb);
        totalFrames++;

        if (ctcHistory.length > HISTORY_LEN) ctcHistory.shift();
        if (sigHistory.length > HISTORY_LEN) sigHistory.shift();
        if (boundHistory.length > HISTORY_LEN) boundHistory.shift();

        // CTC Decoding with Space Reconstruction
        let maxId = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < ctcProbs.length; i++) {
            if (ctcProbs[i] > maxVal) {
                maxVal = ctcProbs[i];
                maxId = i;
            }
        }

        if (maxId !== 0 && maxId !== lastCharId) {
            // Check for inter-word space (class 5)
            if (sigProbs[5] > 0.5 && !decodedText.endsWith(" ")) {
                decodedText += " ";
                decodedEvents.push({char: " ", pos: totalFrames - 1});
            }
            const char = ID_TO_CHAR[maxId];
            if (char) {
                decodedText += char;
                decodedEvents.push({char: char, pos: totalFrames - 1});
                output.textContent = decodedText;
            }
        }
        lastCharId = maxId;
    }

    // Mel History (always 2 frames for the chunk)
    for (let t = 0; t < tLen; t++) {
        melHistory.push(melFrame.slice(t * N_MELS, (t + 1) * N_MELS));
        if (melHistory.length > HISTORY_LEN * 2) melHistory.shift();
    }
    
    // Cleanup decoded events that are out of history
    while (decodedEvents.length > 0 && decodedEvents[0].pos < totalFrames - HISTORY_LEN) {
        decodedEvents.shift();
    }

    drawVisualizations();
    updateDebugInfo(start);

    // 出力テンソルの破棄 (results 内の各テンソル)
        Object.values(results).forEach(t => {
            // ただし、currentStates に代入したものは破棄してはいけない
            // ここでは logits など、状態以外のテンソルを破棄する
            // 実際には、currentStates に代入しなかったものだけを特定して破棄する
            if (t && t.dispose && !Object.values(currentStates).includes(t)) {
                t.dispose();
            }
        });

    } catch (e) {
        console.error("Inference error:", e);
    } finally {
        if (x) x.dispose();
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
    drawProbs(sigCanvas, sigHistory, ["None", "Dit", "Dah", "Intra", "Inter-Char", "Inter-Word"]);
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
    const cellH = h / N_MELS;

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

    // Adaptive normalization for display (matches matplotlib's imshow default)
    let minV = Infinity;
    let maxV = -Infinity;
    for (let t = 0; t < melHistory.length; t++) {
        for (let f = 0; f < N_MELS; f++) {
            const v = melHistory[t][f];
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }
    }
    if (maxV <= minV) maxV = minV + 1;

    for (let t = 0; t < melHistory.length; t++) {
        for (let f = 0; f < N_MELS; f++) {
            const val = (melHistory[t][f] - minV) / (maxV - minV);
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
    if (labels.length === 6) {
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

            const code = MORSE_MAP[char];
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

    // Match data_gen.py: SNR is based on mean power of the signal
    // Standard Morse duty cycle is ~40%. Sine wave power at amp 1.0 is 0.5.
    // So mean signal power is roughly 0.5 * 0.4 = 0.2
    const sigAvgWatts = 0.2;
    const noiseAvgWatts = sigAvgWatts / Math.pow(10, snr / 10);
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
        // Many browsers ignore the requested sampleRate, so we must adapt to what we get
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        await audioContext.audioWorklet.addModule('audio-processor.js');
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    // Always update to actual sample rate
    if (SAMPLE_RATE !== audioContext.sampleRate || !melFilters) {
        SAMPLE_RATE = audioContext.sampleRate;
        melFilters = createMelFilters(N_MELS, N_FFT, SAMPLE_RATE);
        console.log(`AudioContext initialized at ${SAMPLE_RATE}Hz`);
    }
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
    morseNode = new AudioWorkletNode(audioContext, 'morse-processor');
    morseNode.connect(audioContext.destination);
    
    morseNode.port.onmessage = (e) => {
        if (!isRunning) return;
        if (e.data.type === 'audio_chunk') {
            audioBuffer.set(audioBuffer.subarray(HOP_LENGTH));
            audioBuffer.set(e.data.chunk, N_FFT - HOP_LENGTH);
            const mel = applyFFTAndMel(audioBuffer, melFilters);
            melBuffer.push(mel);
            // チャンクサイズを 10フレーム (100ms) に拡大してオーバーヘッドを削減
            if (melBuffer.length >= 10) {
                const combinedMel = new Float32Array(N_MELS * 10);
                for (let i = 0; i < 10; i++) {
                    combinedMel.set(melBuffer[i], i * N_MELS);
                }
                runInference(combinedMel, 10);
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
    decodedText = "";
    output.textContent = "";
    currentStates = initStates();
    melBuffer = [];
    melHistory = [];
    ctcHistory = [];
    sigHistory = [];
    boundHistory = [];
    decodedEvents = [];
    lastCharId = 0;
    totalFrames = 0;
    skipCount = 0;
    drawVisualizations();
    updateDebugInfo();
};

initONNX();