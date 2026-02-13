import { createApp, reactive, ref, onMounted, onUnmounted, watch, nextTick, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';
import { DSP } from '../dsp.js';
import { NoiseNode } from '../noise-node.js';
import { StreamInference } from '../stream-inference.js';
import { MultiStreamManager } from '../multi-stream-manager.js';
import { MORSE_DICT } from '../data_gen.js';
import { getViridisColor, powerToDBNormalized, normalizeDB } from '../visualization.js';
import { PeakDetector } from '../peak-detector.js';

// --- Constants ---
const WINDOW_MS = 32;
const HOP_MS = 10;
const MAX_FREQ = 16000;
const PEAK_LOCK_MS = 3000;
const HISTORY_LEN = 800;
const NOISE_GAIN_VAL = 0.01;
const TARGET_SAMPLE_RATE = 32000;
const LOOKAHEAD_FRAMES = 20; // Sync with inference.js
const HISTORY_LEN_SEC = 15;
const SPECTRAM_HISTORY_LEN = Math.ceil(HISTORY_LEN_SEC * 1000 / HOP_MS);

if (MAX_FREQ > TARGET_SAMPLE_RATE / 2) {
    throw new Error(`MAX_FREQ must be less than TARGET_SAMPLE_RATE/2 (Nyquist frequency). MAX_FREQ=${MAX_FREQ}, TARGET_SAMPLE_RATE=${TARGET_SAMPLE_RATE}`);
}

// --- Station Class for Demo ---
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

        while (this.active) {
            let schedTime = this.ctx.currentTime + 0.1;
            for (const char of text.toUpperCase()) {
                if (!this.active) break;
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
                schedTime += getLen(2);
            }
            await new Promise(r => setTimeout(r, Math.max(0, (schedTime - this.ctx.currentTime) * 1000) + 2000));
        }
    }

    stop() {
        this.active = false;
        try { this.osc.stop(); this.osc.disconnect(); this.gain.disconnect(); } catch (e) { }
    }
}

function calculateVolumeFromSNR(snrDb, sampleRate) {
    return NOISE_GAIN_VAL * Math.sqrt((10000 / sampleRate) * Math.pow(10, snrDb / 10));
}

// --- Vue App ---
const app = createApp({
    setup() {
        // State
        const state = reactive({
            view: 'setup', // 'setup' | 'main'
            decodedText: '',
            trackedFreq: 700,
            isRunning: false,
            isLoading: false,
            loadingProgress: 0,
            loadingStatus: '準備中...',
            detectedPeaks: [], // Latest peaks from PeakDetector
            subDecoders: [],   // [{freq, text, snr, throttled}]
            activeDecoders: 0,
            decoderThrottled: false,
            decoderUtilization: '',
        });

        const decodedTextStyle = computed(() => {
            if (!waterfallOverlayCanvas.value) return {};
            const h = waterfallOverlayCanvas.value.clientHeight;
            const y = h - (state.trackedFreq / MAX_FREQ) * h;
            // Position above the frequency line
            return {
                top: `${y - 45}px`
            };
        });

        // Settings (Persistent)
        const settings = reactive({
            targetFreq: 700,
            volume: 0.3,
            modelPath: '../cw_decoder_quantized.onnx',
            autoTrack: true,
            maxDecoders: 4
        });

        // Load settings from localStorage
        const savedSettings = localStorage.getItem('cw-decoder-settings');
        if (savedSettings) {
            Object.assign(settings, JSON.parse(savedSettings));
            state.trackedFreq = settings.targetFreq;
        }

        // Watch settings to save
        watch(settings, (newSettings) => {
            localStorage.setItem('cw-decoder-settings', JSON.stringify(newSettings));
            if (state.isRunning) {
                // Apply runtime updates
                if (masterGainNode) masterGainNode.gain.setTargetAtTime(newSettings.volume, audioContext.currentTime, 0.1);
                // Force update tracked freq if auto-track is disabled or changed manually
                if (!newSettings.autoTrack && Math.abs(state.trackedFreq - newSettings.targetFreq) > 1) {
                    state.trackedFreq = newSettings.targetFreq;
                    updateUserFilter();
                }
                // Update max decoders at runtime
                if (multiStreamManager) {
                    multiStreamManager.setMaxSlots(newSettings.maxDecoders);
                }
            } else {
                state.trackedFreq = newSettings.targetFreq;
            }
        });

        // Enable ORT Proxy for Worker offloading
        if (typeof ort !== 'undefined' && ort.env) {
            ort.env.wasm.proxy = true;
        }

        const errorMessage = ref('');
        const waterfallCanvas = ref(null);
        const waterfallOverlayCanvas = ref(null);
        const textContainer = ref(null);

        // Audio & Processing State
        let audioContext = null;
        let session = null;
        let streamInference = null;
        let multiStreamManager = null;
        let stream = null;
        let demoNodes = [];
        let masterGainNode = null;
        let userFilterNode = null;
        let waterfallCtx = null;
        let waterfallOverlayCtx = null;

        let windowSize = 0;
        let hopLength = 0;
        let nFft = 0;
        let audioBuf = null;

        let waterfallBuffer = [];
        let peakLockTimer = 0;
        let lastTrackedFreq = 700;
        let rawSpectrumHistory = [];
        let peakDetector = null;


        // --- Core Logic ---

        const initAudioContext = async () => {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await NoiseNode.addModule(audioContext);

                windowSize = Math.floor(TARGET_SAMPLE_RATE * (WINDOW_MS / 1000));
                hopLength = Math.floor(TARGET_SAMPLE_RATE * (HOP_MS / 1000));
                nFft = Math.pow(2, Math.ceil(Math.log2(windowSize)));
                audioBuf = new Float32Array(nFft);

                // Assert consistent bin width (approx 31.25Hz)
                const binWidth = TARGET_SAMPLE_RATE / nFft;
                if (Math.abs(binWidth - 31.25) > 0.1) {
                    console.warn(`Warning: Bin width Changed! ${binWidth} Hz (Expected 31.25 Hz). Model performance may degrade.`);
                }

                await audioContext.audioWorklet.addModule(`../audio-processor.js?t=${Date.now()}`);
            }
            if (audioContext.state === 'suspended') await audioContext.resume();

            if (!peakDetector) {
                const numBins = Math.floor(MAX_FREQ * nFft / TARGET_SAMPLE_RATE);
                peakDetector = new PeakDetector(numBins);
            }
        };

        const fetchModelWithProgress = async (url, onProgress) => {
            const response = await fetch(url);
            const contentLength = response.headers.get('content-length');
            const total = parseInt(contentLength, 10);
            const reader = response.body.getReader();
            let received = 0;
            const chunks = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                received += value.length;
                if (total) {
                    onProgress((received / total) * 100);
                }
            }

            const allChunks = new Uint8Array(received);
            let position = 0;
            for (const chunk of chunks) {
                allChunks.set(chunk, position);
                position += chunk.length;
            }
            return allChunks.buffer; // Return ArrayBuffer
        };

        const initModel = async () => {
            if (multiStreamManager) {
                multiStreamManager.dispose();
                multiStreamManager = null;
            }
            if (streamInference) {
                streamInference.dispose();
                streamInference = null;
            }

            // Enable WebAssembly Proxy (requires ort-wasm-simd-threaded.jsep.wasm)
            // This moves the heavy inference loop to a worker
            ort.env.wasm.proxy = true;

            try {
                state.loadingStatus = 'モデルをダウンロード中...';
                state.loadingProgress = 0;

                const modelBuffer = await fetchModelWithProgress(settings.modelPath, (progress) => {
                    state.loadingProgress = progress;
                });

                state.loadingStatus = 'AIエンジンを初期化中...';

                // Force UI update before heavy initialization
                await new Promise(r => setTimeout(r, 10));

                session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: ['wasm']
                });

                // MultiStreamManager (複数ピーク同時デコード)
                multiStreamManager = new MultiStreamManager(session, ort, {
                    maxSlots: settings.maxDecoders,
                    chunkSize: 12,
                    hopMs: HOP_MS,
                    useWebGPU: false,
                    historyLength: HISTORY_LEN
                });

                // メインスロットの StreamInference への互換参照
                // (drawWaterfallOverlay 等で使用)
                streamInference = null; // mainInference は multiStreamManager 経由で取得

            } catch (e) {
                throw e;
            }
        };

        const updateUserFilter = () => {
            if (userFilterNode && Math.abs(state.trackedFreq - lastTrackedFreq) > 10) {
                userFilterNode.frequency.setTargetAtTime(state.trackedFreq, audioContext.currentTime, 0.1);
                lastTrackedFreq = state.trackedFreq;
            }
        };

        const runPeakDetection = (magnitudes) => {
            if (!peakDetector) return;

            const peaks = peakDetector.process(magnitudes);

            const freqPeaks = peaks.map(p => ({
                f: p.index * TARGET_SAMPLE_RATE / nFft,
                p: p.magnitude,
                snr: p.snr
            }));

            // Store detected peaks for visualization (always, even if not tracking)
            state.detectedPeaks = freqPeaks;

            if (settings.autoTrack) {
                const now = Date.now();
                // Find peak nearby current tracked frequency (within ±50Hz)
                const nearbyPeak = freqPeaks.find(p => Math.abs(p.f - state.trackedFreq) < 50);

                if (nearbyPeak) {
                    // Smoothly track the nearby peak
                    state.trackedFreq = state.trackedFreq * 0.9 + nearbyPeak.f * 0.1;
                    peakLockTimer = now + PEAK_LOCK_MS;
                } else if (now > peakLockTimer && freqPeaks.length > 0) {
                    // If locked lost for a while, snap to the strongest peak
                    state.trackedFreq = freqPeaks[0].f;
                    peakLockTimer = now + PEAK_LOCK_MS;
                }
            } else {
                // Smooth transition to manual target
                state.trackedFreq = state.trackedFreq * 0.9 + settings.targetFreq * 0.1;
            }
            updateUserFilter();

            // MultiStreamManager にピーク情報を伝搬
            if (multiStreamManager) {
                multiStreamManager.updatePeaks(freqPeaks, state.trackedFreq);
            }
        };

        const processAudioChunk = (chunk) => {
            audioBuf.set(audioBuf.subarray(chunk.length));
            audioBuf.set(chunk, nFft - chunk.length);

            const real = new Float32Array(nFft);
            real.set(audioBuf.subarray(nFft - windowSize));
            const imag = new Float32Array(nFft);
            for (let j = 0; j < windowSize; j++) {
                real[j] *= 0.5 * (1 - Math.cos(2 * Math.PI * j / windowSize));
            }
            DSP.fft(real, imag);

            const numBins = Math.floor(MAX_FREQ * nFft / TARGET_SAMPLE_RATE);
            const magnitudes = new Float32Array(numBins);
            for (let j = 0; j < numBins; j++) magnitudes[j] = real[j] * real[j] + imag[j] * imag[j];

            runPeakDetection(magnitudes);

            // Store raw magnitudes for re-decoding
            rawSpectrumHistory.push(magnitudes);
            if (rawSpectrumHistory.length > SPECTRAM_HISTORY_LEN) rawSpectrumHistory.shift();

            // MultiStreamManager: 全スロットにmagnitudesを配信
            if (multiStreamManager) {
                multiStreamManager.pushMagnitudes(magnitudes, nFft, TARGET_SAMPLE_RATE);

                // メインスロットのテキストを更新
                state.decodedText = multiStreamManager.mainText;
                if (textContainer.value) {
                    textContainer.value.scrollLeft = textContainer.value.scrollWidth;
                }

                // サブデコーダー情報を更新
                state.subDecoders = multiStreamManager.getSubSlots();
                const stats = multiStreamManager.performanceStats;
                state.activeDecoders = stats.activeSlots;
                state.decoderThrottled = stats.throttled;
                state.decoderUtilization = `${stats.totalInferenceTime.toFixed(0)}/${stats.budget.toFixed(0)}=${(stats.utilization * 100).toFixed(0)}%`;
            }

            const displayMagnitude = magnitudes.map(p => Math.max(0, Math.log1p(p * 5000) / 12));
            waterfallBuffer.push(displayMagnitude);
            if (waterfallBuffer.length > HISTORY_LEN) waterfallBuffer.shift();
            if (waterfallBuffer.length > HISTORY_LEN) waterfallBuffer.shift();
        };

        const redecode = async () => {
            if (!multiStreamManager || rawSpectrumHistory.length === 0) return;

            // メインスロットのみリデコード
            multiStreamManager.reset();
            if (peakDetector) peakDetector.reset();
            state.decodedText = '';

            // メインスロットを再作成
            multiStreamManager.updatePeaks([], state.trackedFreq);

            for (const magnitudes of rawSpectrumHistory) {
                multiStreamManager.pushMagnitudes(magnitudes, nFft, TARGET_SAMPLE_RATE);
            }
            state.decodedText = multiStreamManager.mainText;
        };

        const setupProcessing = (source) => {
            if (audioContext.state === 'suspended') audioContext.resume();

            // Ensure previous worklet is cleaned up
            // Note: demoNodes cleanup is handled in stop/start but let's be safe

            const workletNode = new AudioWorkletNode(audioContext, 'morse-processor', {
                processorOptions: {
                    hopLength: hopLength,
                    targetSampleRate: TARGET_SAMPLE_RATE
                }
            });
            workletNode.port.onmessage = (e) => {
                if (state.isRunning && e.data.type === 'audio_chunk') processAudioChunk(e.data.chunk);
            };
            source.connect(workletNode);
            demoNodes.push(workletNode);
        };

        const startMic = async () => {
            if (state.isLoading) return;
            state.isLoading = true;
            errorMessage.value = '';
            try {
                await initAudioContext();
                await initModel();

                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: { ideal: 2, min: 1 },
                        echoCancellation: { exact: false },
                        noiseSuppression: { exact: false },
                        autoGainControl: { exact: false },
                    }
                });
                const source = audioContext.createMediaStreamSource(stream);

                // Clear previous session data
                waterfallBuffer = [];
                state.detectedPeaks = [];
                rawSpectrumHistory = [];
                state.decodedText = '';
                if (multiStreamManager) multiStreamManager.reset();
                if (peakDetector) peakDetector.reset();

                setupProcessing(source);

                userFilterNode = audioContext.createBiquadFilter();
                userFilterNode.type = 'bandpass';
                userFilterNode.frequency.value = state.trackedFreq;
                userFilterNode.Q.value = 10;

                masterGainNode = audioContext.createGain();
                masterGainNode.gain.value = settings.volume;

                source.connect(userFilterNode);
                userFilterNode.connect(masterGainNode);
                masterGainNode.connect(audioContext.destination);

                state.isRunning = true;
                state.view = 'main';

                nextTick(() => {
                    initCanvas();
                    requestAnimationFrame(drawLoop);
                });
            } catch (e) {
                errorMessage.value = "マイクアクセス失敗: " + e;
                console.error(e);
            } finally {
                state.isLoading = false;
            }
        };

        const startDemo = async () => {
            if (state.isLoading) return;
            state.isLoading = true;
            errorMessage.value = '';
            try {
                await initAudioContext();
                await initModel();

                const analysisMix = audioContext.createGain();
                masterGainNode = audioContext.createGain();
                masterGainNode.gain.value = settings.volume;

                userFilterNode = audioContext.createBiquadFilter();
                userFilterNode.type = 'bandpass';
                userFilterNode.frequency.value = state.trackedFreq;
                userFilterNode.Q.value = 10;

                masterGainNode.connect(userFilterNode);
                userFilterNode.connect(audioContext.destination);

                const noise = new NoiseNode(audioContext, { type: 'whitenoise' });
                const noiseGain = audioContext.createGain();
                noiseGain.gain.value = NOISE_GAIN_VAL;
                noise.connect(noiseGain);
                noiseGain.connect(analysisMix);
                noiseGain.connect(masterGainNode);
                demoNodes.push(noise);

                const stations = [
                    { freq: 650, wpm: 18, jitter: 0.05, snr: 20, msg: "CQ CQ DE JA1ABC K" },
                    { freq: 800, wpm: 20, jitter: 0.15, snr: -5, msg: "CQ CQ DE JH1XYZ K" },
                    { freq: 1200, wpm: 25, jitter: 0.1, snr: 10, msg: "CQ CQ DE K1XYZ K" },
                    { freq: 1800, wpm: 35, jitter: 0.02, snr: 30, msg: "CQ CQ DE G4ZOO K" },
                    { freq: 2500, wpm: 20, jitter: 0.15, snr: 0, msg: "CQ CQ DE JH1UMV K" },
                    { freq: 2800, wpm: 12, jitter: 0.15, snr: -10, msg: "CQ CQ DE JX1KLM K" },
                    { freq: 3200, wpm: 28, jitter: 0.05, snr: 15, msg: "CQ CQ DE DF7CB K" },
                    { freq: 4000, wpm: 28, jitter: 0.05, snr: 15, msg: "CQ CQ DE JA7ABC K" },
                    { freq: 5200, wpm: 28, jitter: 0.05, snr: 15, msg: "CQ CQ DE JQ1XYZ K" },
                    { freq: 6000, wpm: 28, jitter: 0.05, snr: 15, msg: "CQ CQ DE JC8ABC K" },
                    { freq: 7500, wpm: 28, jitter: 0.05, snr: 15, msg: "CQ CQ DE JX2KLM K" },
                    { freq: 8200, wpm: 22, jitter: 0.05, snr: 15, msg: "CQ CQ DE JA1HJK K" },
                    { freq: 9500, wpm: 25, jitter: 0.05, snr: 10, msg: "CQ CQ DE K1AW K" },
                    { freq: 11500, wpm: 20, jitter: 0.1, snr: 5, msg: "CQ CQ DE G3ZZZ K" },
                    { freq: 13000, wpm: 28, jitter: 0.05, snr: 20, msg: "CQ CQ DE VK2FGH K" },
                    { freq: 14000, wpm: 15, jitter: 0.1, snr: 0, msg: "CQ CQ DE ZL1JKL K" }
                ];

                stations.forEach(s => {
                    const vol = calculateVolumeFromSNR(s.snr, audioContext.sampleRate);
                    const st = new Station(audioContext, s.freq, s.wpm, s.jitter, vol, analysisMix);
                    st.gain.connect(masterGainNode);
                    st.play(s.msg);
                    demoNodes.push(st);
                });

                // Clear previous session data
                waterfallBuffer = [];
                state.detectedPeaks = [];
                rawSpectrumHistory = [];
                state.decodedText = '';
                if (multiStreamManager) multiStreamManager.reset();
                if (peakDetector) peakDetector.reset();

                setupProcessing(analysisMix);

                state.isRunning = true;
                state.view = 'main';

                nextTick(() => {
                    initCanvas();
                    requestAnimationFrame(drawLoop);
                });
            } catch (e) {
                errorMessage.value = "デモ開始失敗: " + e;
                console.error(e);
            } finally {
                state.isLoading = false;
            }
        };

        const stop = () => {
            state.isRunning = false;
            if (multiStreamManager) multiStreamManager.reset();
            if (peakDetector) peakDetector.reset();

            if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
            demoNodes.forEach(n => { if (n.stop) n.stop(); if (n.disconnect) n.disconnect(); });
            demoNodes = [];
            masterGainNode = null;
            if (userFilterNode) { userFilterNode.disconnect(); userFilterNode = null; }

            state.view = 'setup';
        };

        const adjustFreq = (delta) => {
            settings.targetFreq = Math.max(100, Math.min(3000, settings.targetFreq + delta));
            // Also update trackedFreq immediately in setup mode
            state.trackedFreq = settings.targetFreq;
            redecode();
        };

        const toggleSettings = () => {
            stop(); // For now, simple toggle means stop and go back to setup
        };

        const clearCache = async () => {
            if (!confirm('キャッシュを削除してリロードしますか？')) return;
            try {
                if ('caches' in window) {
                    const keys = await caches.keys();
                    await Promise.all(keys.map(key => caches.delete(key)));
                }
                // Unregister SW
                if ('serviceWorker' in navigator) {
                    const regs = await navigator.serviceWorker.getRegistrations();
                    await Promise.all(regs.map(reg => reg.unregister()));
                }
                window.location.reload();
            } catch (e) {
                alert('キャッシュ削除失敗: ' + e);
            }
        };

        // --- Visualization ---

        const initCanvas = () => {
            if (waterfallCanvas.value) {
                waterfallCanvas.value.width = waterfallCanvas.value.clientWidth;
                waterfallCanvas.value.height = waterfallCanvas.value.clientHeight;
                waterfallCtx = waterfallCanvas.value.getContext('2d', { alpha: false });
            }
            if (waterfallOverlayCanvas.value) {
                waterfallOverlayCanvas.value.width = waterfallOverlayCanvas.value.clientWidth;
                waterfallOverlayCanvas.value.height = waterfallOverlayCanvas.value.clientHeight;
                waterfallOverlayCtx = waterfallOverlayCanvas.value.getContext('2d');
            }
        };

        // Handle resize
        window.addEventListener('resize', () => {
            if (state.view === 'main') initCanvas();
        });

        // Touch to set frequency on Waterfall
        const handleTouch = (e) => {
            // Use overlay canvas for interaction if available, else fallback
            const targetCanvas = waterfallOverlayCanvas.value || waterfallCanvas.value;
            if (state.view !== 'main' || !targetCanvas) return;
            const rect = targetCanvas.getBoundingClientRect();
            const y = e.touches ? e.touches[0].clientY : e.clientY;

            const relY = y - rect.top;

            // Waterfall draws frequencies from bottom (0) to top (MAX_FREQ)
            const h = targetCanvas.height;
            const freq = (1 - relY / h) * MAX_FREQ;

            settings.targetFreq = freq;
            settings.autoTrack = false; // Disable auto track on manual touch
            state.trackedFreq = freq;
            peakLockTimer = Date.now() + PEAK_LOCK_MS;
            updateUserFilter();
            redecode();
        };

        onMounted(() => {
        });

        watch(waterfallOverlayCanvas, (el) => {
            if (el) {
                el.addEventListener('pointerdown', handleTouch);
                el.addEventListener('pointermove', (e) => {
                    if (e.buttons > 0) handleTouch(e);
                });
            }
        });

        const drawLoop = () => {
            if (!state.isRunning) return;

            drawWaterfall();
            drawWaterfallOverlay();

            requestAnimationFrame(drawLoop);
        };

        const drawWaterfall = () => {
            if (!waterfallCtx || waterfallBuffer.length === 0) return;

            const w = waterfallCanvas.value.width;
            const h = waterfallCanvas.value.height;
            const numFrames = waterfallBuffer.length;

            // Shift
            waterfallCtx.drawImage(waterfallCanvas.value, -numFrames, 0);

            for (let f = 0; f < numFrames; f++) {
                const frameData = waterfallBuffer[f];
                const numBins = frameData.length;
                const binH = h / numBins;
                const x = w - numFrames + f;

                for (let i = 0; i < numBins; i++) {
                    const [r, g, b] = getViridisColor(frameData[i]);
                    waterfallCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    // Draw from bottom up
                    waterfallCtx.fillRect(x, h - (i + 1) * binH, 1, binH + 1);
                }
            }
            waterfallBuffer = []; // Clear buffer
        };

        const drawWaterfallOverlay = () => {
            const mainInference = multiStreamManager ? multiStreamManager.mainInference : null;
            if (!waterfallOverlayCtx || !mainInference) return;
            const w = waterfallOverlayCanvas.value.width;
            const h = waterfallOverlayCanvas.value.height;

            waterfallOverlayCtx.clearRect(0, 0, w, h);

            // 1. Calculate Metrics for Layering
            const binBW = TARGET_SAMPLE_RATE / nFft;
            const focusBW = 14 * binBW;
            const focusH = (focusBW / MAX_FREQ) * h;
            const trackY = h - (state.trackedFreq / MAX_FREQ) * h;

            // --- LAYER 1: Focus Band Overlay (Semi-transparent black) ---
            waterfallOverlayCtx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            waterfallOverlayCtx.fillRect(0, trackY - focusH / 2, w, focusH);

            // --- LAYER 2: Target Line & Inference History (Dit/Dah Bars) ---
            waterfallOverlayCtx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            const barW = 80;
            waterfallOverlayCtx.fillRect(w - barW, trackY - 0.5, barW, 1);

            const sigHistory = mainInference.getSignalHistory();
            const eventHistory = mainInference.getEvents();
            const totalFrames = mainInference.frameCount;

            if (sigHistory.length > 0) {
                const barH = 10;
                const sigColors = [
                    'rgba(0,0,0,0)',
                    'rgba(255,60,60,0.7)',
                    'rgba(60,60,255,0.7)',
                    'rgba(0, 0, 0, 0.3)',
                ];

                sigHistory.forEach(item => {
                    const x = w - (totalFrames - item.pos) - LOOKAHEAD_FRAMES;
                    if (x < 0 || x >= w) return;

                    let maxIdx = 0, maxP = -1;
                    for (let s = 0; s < 4; s++) {
                        if (item.probs[s] > maxP) { maxP = item.probs[s]; maxIdx = s; }
                    }

                    if (maxIdx > 0) {
                        waterfallOverlayCtx.fillStyle = sigColors[maxIdx];
                        // Centered on the track line
                        waterfallOverlayCtx.fillRect(x - 1, trackY - barH / 2, 2, barH);
                    }
                });
            }

            // --- LAYER 3: Boundary Markers (Red Dots) & Decoded Characters ---
            // Positioned BELOW the center line (trackY)
            const markerY = trackY + 8;
            const charY = trackY + 25;

            eventHistory.forEach(ev => {
                const x = w - (totalFrames - ev.pos) - LOOKAHEAD_FRAMES;
                if (x > 0 && x < w) {
                    // Red Dot Marker
                    waterfallOverlayCtx.fillStyle = '#ff0000';
                    waterfallOverlayCtx.beginPath();
                    waterfallOverlayCtx.arc(x, markerY, 3, 0, Math.PI * 2);
                    waterfallOverlayCtx.fill();

                    // Character text
                    waterfallOverlayCtx.fillStyle = '#fff';
                    waterfallOverlayCtx.font = 'bold 16px Courier New';
                    waterfallOverlayCtx.textAlign = 'center';
                    const displayChar = ev.char === ' ' ? '\u2423' : ev.char;
                    waterfallOverlayCtx.fillText(displayChar, x, charY);
                }
            });

            // --- LAYER 4: Peak Detector SNR Labels (ON TOP) ---
            const snrOffset = 10 * Math.log10(binBW / 2500);
            // "Locked" indicator at the very edge (Topmost)
            waterfallOverlayCtx.fillStyle = settings.autoTrack ? '#f00' : '#888';
            waterfallOverlayCtx.fillRect(w - 5, trackY - 5, 5, 10);
        };

        const getPeakY = (freq) => {
            if (!waterfallCanvas.value) return 0;
            const h = waterfallCanvas.value.height;
            return h * (1 - freq / MAX_FREQ);
        };

        const getPeakSNR = (snr) => {
            const binBW = TARGET_SAMPLE_RATE / nFft;
            const snrOffset = 10 * Math.log10(binBW / 2500);
            return 10 * Math.log10(snr + 1e-12) + snrOffset;
        };

        const selectPeak = (freq) => {
            settings.targetFreq = freq;
            settings.autoTrack = false; // Disable auto tracking on manual peak selection
            state.trackedFreq = freq;
            peakLockTimer = Date.now() + PEAK_LOCK_MS;
            updateUserFilter();
            redecode();
        };

        const getSubDecoderText = (freq) => {
            if (!multiStreamManager) return null;
            return multiStreamManager.getTextForFreq(freq);
        };

        // detectedPeaks とアクティブサブスロットをマージ
        // ピーク消失後もスロットが生存中ならマーカーを残す
        const mergedPeaks = computed(() => {
            const NEARBY = 50;
            // state.subDecoders はリアクティブ（processAudioChunk で毎フレーム更新）
            const subSlots = state.subDecoders;
            const usedSlotFreqs = new Set();

            const result = state.detectedPeaks.map(p => {
                // このピークに対応するスロットを探す
                const matchedSlot = subSlots.find(s => Math.abs(s.freq - p.f) < NEARBY);
                if (matchedSlot) usedSlotFreqs.add(matchedSlot.freq);
                return {
                    f: p.f, p: p.p, snr: p.snr,
                    hasSlot: !!matchedSlot,
                    slotText: matchedSlot ? matchedSlot.text : '',
                };
            });

            // スロットはあるが detected peaks にないものを追加
            for (const sub of subSlots) {
                if (sub.throttled) continue;
                if (usedSlotFreqs.has(sub.freq)) continue;
                const alreadyShown = result.some(r => Math.abs(r.f - sub.freq) < NEARBY);
                if (!alreadyShown && sub.text) {
                    result.push({
                        f: sub.freq,
                        p: 0,
                        snr: sub.snr,
                        hasSlot: true,
                        slotText: sub.text,
                    });
                }
            }

            return result;
        });

        return {
            state,
            settings,
            errorMessage,
            decodedTextStyle,
            startMic,
            startDemo,
            stop,
            adjustFreq,
            toggleSettings,
            clearCache,
            waterfallCanvas,
            waterfallOverlayCanvas,
            textContainer,
            getPeakY,
            getPeakSNR,
            selectPeak,
            getSubDecoderText,
            mergedPeaks
        };
    }
}).mount('#app');