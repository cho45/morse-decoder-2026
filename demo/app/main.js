import { createApp, reactive, ref, onMounted, onUnmounted, watch, nextTick, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';
import { DSP } from '../dsp.js';
import { NoiseNode } from '../noise-node.js';
import { StreamInference } from '../stream-inference.js';
import { MORSE_DICT } from '../data_gen.js';
import { getViridisColor, powerToDBNormalized, normalizeDB } from '../visualization.js';
import { PeakDetector } from '../peak-detector.js';

// --- Constants ---
const WINDOW_MS = 32;
const HOP_MS = 10;
const MAX_FREQ = 4000;
const PEAK_LOCK_MS = 3000;
const HISTORY_LEN = 800;
const NOISE_GAIN_VAL = 0.01;
const TARGET_SAMPLE_RATE = 16000;
const LOOKAHEAD_FRAMES = 20; // Sync with inference.js
const HISTORY_LEN_SEC = 15;
const SPECTRAM_HISTORY_LEN = Math.ceil(HISTORY_LEN_SEC * 1000 / HOP_MS);

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
            detectedPeaks: [], // Latest peaks from PeakDetector
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
            autoTrack: true
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

                await audioContext.audioWorklet.addModule('../audio-processor.js');
            }
            if (audioContext.state === 'suspended') await audioContext.resume();

            if (!peakDetector) {
                const numBins = Math.floor(MAX_FREQ * nFft / TARGET_SAMPLE_RATE);
                peakDetector = new PeakDetector(numBins);
            }
        };

        const initModel = async () => {
            if (streamInference) {
                streamInference.dispose();
                streamInference = null;
            }

            // Enable WebAssembly Proxy (requires ort-wasm-simd-threaded.jsep.wasm)
            // This moves the heavy inference loop to a worker
            ort.env.wasm.proxy = true;

            session = await ort.InferenceSession.create(settings.modelPath, {
                executionProviders: ['wasm']
            });
            streamInference = new StreamInference(session, ort, {
                chunkSize: 12,
                useWebGPU: false,
                historyLength: HISTORY_LEN
            });
            streamInference.addEventListener('result', (e) => {
                state.decodedText = e.detail.text;
                // Auto scroll text
                if (textContainer.value) {
                    textContainer.value.scrollLeft = textContainer.value.scrollWidth;
                }
            });
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


            // Extract 14 bins
            const binBW = TARGET_SAMPLE_RATE / nFft;
            const W = 14 * binBW;
            const fStart = state.trackedFreq - W / 2 + binBW / 2;
            const specFrame = new Float32Array(14);
            for (let i = 0; i < 14; i++) {
                const f = fStart + i * binBW;
                const k = f * nFft / TARGET_SAMPLE_RATE;
                const kIdx = Math.floor(k);
                const kFrac = k - kIdx;
                if (kIdx >= 0 && kIdx < nFft - 1) {
                    const p1 = real[kIdx] * real[kIdx] + imag[kIdx] * imag[kIdx];
                    const p2 = real[kIdx + 1] * real[kIdx + 1] + imag[kIdx + 1] * imag[kIdx + 1];
                    specFrame[i] = p1 * (1 - kFrac) + p2 * kFrac;
                }
            }

            if (streamInference) streamInference.pushFrame(specFrame);

            const displayMagnitude = magnitudes.map(p => Math.max(0, Math.log1p(p * 5000) / 12));
            waterfallBuffer.push(displayMagnitude);
            if (waterfallBuffer.length > HISTORY_LEN) waterfallBuffer.shift();
            if (waterfallBuffer.length > HISTORY_LEN) waterfallBuffer.shift();
        };

        const redecode = async () => {
            if (!streamInference || rawSpectrumHistory.length === 0) return;

            streamInference.reset();
            if (peakDetector) peakDetector.reset();
            state.decodedText = '';

            const binBW = TARGET_SAMPLE_RATE / nFft;
            const W = 14 * binBW;
            const fStart = state.trackedFreq - W / 2 + binBW / 2;

            for (const magnitudes of rawSpectrumHistory) {
                const specFrame = new Float32Array(14);
                for (let i = 0; i < 14; i++) {
                    const f = fStart + i * binBW;
                    const k = f * nFft / TARGET_SAMPLE_RATE;
                    const kIdx = Math.floor(k);
                    const kFrac = k - kIdx;
                    if (kIdx >= 0 && kIdx < magnitudes.length - 1) {
                        const p1 = magnitudes[kIdx];
                        const p2 = magnitudes[kIdx + 1];
                        specFrame[i] = p1 * (1 - kFrac) + p2 * kFrac;
                    }
                }
                streamInference.pushFrame(specFrame);
            }
        };

        const setupProcessing = (source) => {
            const workletNode = new AudioWorkletNode(audioContext, 'morse-processor', {
                processorOptions: {
                    sampleRate: audioContext.sampleRate,
                    hopLength: hopLength
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
                    { freq: 1200, wpm: 25, jitter: 0.1, snr: 10, msg: "CQ CQ DE K1XYZ K" },
                    { freq: 1800, wpm: 35, jitter: 0.02, snr: 30, msg: "CQ CQ DE G4ZOO K" },
                    { freq: 2500, wpm: 20, jitter: 0.15, snr: 0, msg: "CQ CQ DE JH1UMV K" },
                    { freq: 800, wpm: 20, jitter: 0.15, snr: -5, msg: "CQ CQ DE JH1XYZ K" },
                    { freq: 3200, wpm: 28, jitter: 0.05, snr: 15, msg: "CQ CQ DE DF7CB K" }
                ];

                stations.forEach(s => {
                    const vol = calculateVolumeFromSNR(s.snr, audioContext.sampleRate);
                    const st = new Station(audioContext, s.freq, s.wpm, s.jitter, vol, analysisMix);
                    st.gain.connect(masterGainNode);
                    st.play(s.msg);
                    demoNodes.push(st);
                });

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
            if (streamInference) streamInference.reset();
            if (peakDetector) peakDetector.reset();

            waterfallBuffer = [];
            state.detectedPeaks = [];

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
            if (!waterfallOverlayCtx || !streamInference) return;
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

            const sigHistory = streamInference.getSignalHistory();
            const eventHistory = streamInference.getEvents();
            const totalFrames = streamInference.frameCount;

            if (sigHistory.length > 0) {
                const barH = 10;
                const sigColors = [
                    'rgba(0,0,0,0)',
                    'rgba(255,0,0,0.5)',
                    'rgba(0,0,255,0.5)',
                    'rgba(255, 255, 255, 0.3)',
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
            state.detectedPeaks.forEach((peak) => {
                const y = h - (peak.f / MAX_FREQ) * h;
                const snrDb = 10 * Math.log10(peak.snr + 1e-12) + snrOffset;

                // Marker: Red with White border
                waterfallOverlayCtx.fillStyle = 'red';
                waterfallOverlayCtx.strokeStyle = 'white';
                waterfallOverlayCtx.lineWidth = 1.5;

                waterfallOverlayCtx.beginPath();
                waterfallOverlayCtx.arc(w - 10, y, 4, 0, Math.PI * 2);
                waterfallOverlayCtx.fill();
                waterfallOverlayCtx.stroke();

                // Label: Normalized SNR
                waterfallOverlayCtx.fillStyle = 'white';
                waterfallOverlayCtx.font = 'bold 11px sans-serif';
                waterfallOverlayCtx.textAlign = 'right';
                waterfallOverlayCtx.fillText(snrDb.toFixed(1) + ' dB', w - 20, y - 6);
            });

            // "Locked" indicator at the very edge (Topmost)
            waterfallOverlayCtx.fillStyle = settings.autoTrack ? '#f00' : '#888';
            waterfallOverlayCtx.fillRect(w - 5, trackY - 5, 5, 10);
        };

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
            textContainer
        };
    }
}).mount('#app');