import { N_FFT, N_BINS, HOP_LENGTH, computeSpecFrames } from '../inference.js';
import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";

// Configuration
const NUM_WORKERS = 4;
const MODEL_PATH = "../cw_decoder_quantized.onnx";
const DURATION_SEC = 2; // 2 seconds of audio per request

// State
let workers = [];
let audioBuffer = null;
let specFrames = null;
let modelBuffer = null;

const logEl = document.getElementById("log");
const startBtn = document.getElementById("start-btn");
const container = document.getElementById("worker-container");

function log(msg) {
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
    console.log(msg);
}

function createWorkerCard(id) {
    const div = document.createElement("div");
    div.className = "worker-card";
    div.innerHTML = `
        <div class="worker-id">Worker #${id}</div>
        <div class="worker-status" id="status-${id}">Initializing...</div>
    `;
    return div;
}

function updateStatus(id, status) {
    const el = document.getElementById(`status-${id}`);
    if (el) el.innerText = status;
}

let readyCount = 0;
function checkAllReady() {
    readyCount++;
    if (readyCount === NUM_WORKERS) {
        log("All workers ready.");
        startBtn.disabled = false;
    }
}

async function init() {
    log("Initializing PoC (Main-thread Model Load)...");

    // UI Setup
    for (let i = 0; i < NUM_WORKERS; i++) {
        container.appendChild(createWorkerCard(i));
    }

    try {
        log(`Fetching model from ${MODEL_PATH}...`);
        const response = await fetch(MODEL_PATH);
        modelBuffer = await response.arrayBuffer();
        log(`Model loaded. Size: ${(modelBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

        // Generate dummy audio
        const sampleRate = 16000;
        const length = sampleRate * DURATION_SEC;
        audioBuffer = new Float32Array(length);
        for (let i = 0; i < length; i++) audioBuffer[i] = (Math.random() * 2 - 1) * 0.1;

        log("Computing spectrogram for test data...");
        specFrames = computeSpecFrames(audioBuffer, sampleRate);
        log(`Spectrogram computed. Shape: [${specFrames.length / N_BINS}, ${N_BINS}]`);

        // Initialize Workers
        for (let i = 0; i < NUM_WORKERS; i++) {
            const w = new Worker("worker-poc.js", { type: "module" });
            const api = Comlink.wrap(w);
            workers.push({ api, id: i });

            // Send a COPY of the model buffer to each worker
            // Since ArrayBuffer is transferable, we must copy it to keep it in the main thread
            // or send it to other workers.
            const copy = modelBuffer.slice(0);
            api.init(Comlink.transfer(copy, [copy])).then(() => {
                log(`Worker ${i} initialized.`);
                updateStatus(i, "Ready");
                checkAllReady();
            }).catch(e => {
                log(`Worker ${i} init error: ${e.message}`);
                console.error(e);
            });
        }
    } catch (e) {
        log(`Error during init: ${e}`);
        console.error(e);
    }
}

startBtn.addEventListener("click", () => {
    log("Starting parallel inference...");
    startBtn.disabled = true;

    const numFrames = specFrames.length / N_BINS;
    const validFrames = Math.floor(numFrames / 4) * 4;
    log(`Trimming frames from ${numFrames} to ${validFrames} for compatibility.`);

    workers.forEach(async ({ api, id }) => {
        updateStatus(id, "Running...");
        const chunk = specFrames.slice(0, validFrames * N_BINS); // Copy & Trim

        try {
            const start = performance.now();
            const result = await api.infer(Comlink.transfer(chunk, [chunk.buffer]));
            const end = performance.now();

            log(`Worker ${id} infer done (Worker: ${result.time.toFixed(1)}ms, Total: ${(end - start).toFixed(1)}ms). Logits: ${result.logits.length}`);
            updateStatus(id, "Done");
        } catch (e) {
            log(`Worker ${id} error: ${e}`);
            console.error(e);
            updateStatus(id, "Error");
        }
    });

    setTimeout(() => { startBtn.disabled = false; }, 1000); // Simple debounce
});

init();
