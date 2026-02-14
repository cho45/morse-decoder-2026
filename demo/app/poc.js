import { N_BINS } from '../inference.js';
import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";

const NUM_WORKERS = 4;
const SLOTS_PER_WORKER = 4;
const MODEL_PATH = "../cw_decoder_quantized.onnx";
const CHUNK_SIZE = 12;
const INTERVAL_MS = 120;
const MIN_BUSY_MS = 80;

/**
 * SlotProxy - PoC独自のUI用クラス
 *
 * 責務:
 * - ワーカー内のスロットとの通信
 * - UI要素（ビジーインジケーター、ラベル、テキスト）の更新
 * - ビジー状態の管理
 *
 * 注: 実際のアプリケーション (demo/app/main.js) には対応するものがない。
 *     main.js は Vue 3 を使用し、リアクティブステートでUIを管理している。
 */
class SlotProxy {
    /**
     * @param {Object} workerApi - ワーカーAPI
     * @param {string} slotId - スロットID
     */
    constructor(workerApi, slotId) {
        this.api = workerApi;
        this.slotId = slotId;
        this.indicator = null;
        this.labelEl = null;
        this.textEl = null;
        this.lastBusyTime = 0;
        this.isBusy = false;
    }

    /**
     * ワーカーに接続し、結果イベントを購読する
     * @returns {Promise<void>}
     */
    async connect() {
        await this.api.subscribe(
            this.slotId,
            Comlink.proxy(this.handleResult.bind(this))
        );
    }

    /**
     * ビジー状態を更新する
     * @param {boolean} isBusy
     */
    handleStatus(isBusy) {
        this.isBusy = isBusy;
        if (this.isBusy) {
            this.lastBusyTime = Date.now();
        }
        this.render();
    }

    /**
     * 結果イベントを処理する
     * @param {Object} detail
     */
    handleResult(detail) {
        if (this.textEl) {
            this.textEl.innerText = detail.text;
        }
    }

    /**
     * UIを更新する
     */
    render() {
        if (!this.indicator || !this.labelEl) return;

        const now = Date.now();
        const shouldShowBusy = this.isBusy || (now - this.lastBusyTime < MIN_BUSY_MS);

        if (shouldShowBusy) {
            this.indicator.className = "busy-indicator busy";
            this.labelEl.innerText = "Busy";
        } else {
            this.indicator.className = "busy-indicator idle";
            this.labelEl.innerText = "Ready";
        }

        // Sticky timer
        if (!this.isBusy && (now - this.lastBusyTime < MIN_BUSY_MS)) {
            setTimeout(() => this.render(), MIN_BUSY_MS - (now - this.lastBusyTime) + 1);
        }
    }

    /**
     * フレームを処理する
     * @param {Float32Array[]} frames
     * @returns {Promise<void>}
     */
    async pushFrames(frames) {
        this.handleStatus(true);
        try {
            await this.api.pushFrames(this.slotId, frames);
        } finally {
            this.handleStatus(false);
        }
    }
}

/**
 * MultiStreamProxy - 複数ワーカーを管理するクラス
 *
 * 責務:
 * - 複数のワーカーを初期化・管理する
 * - ワーカーごとのスロットを作成・管理する
 * - 全スロットにフレームを配信する
 * - パフォーマンス統計を収集する
 *
 * 注: 実際の実装で使えるレベルの抽象化レイヤー。
 *     UIに依存しない設計。
 */
class MultiStreamProxy {
    /**
     * @param {number} numWorkers - ワーカー数
     * @param {Object} options - 設定オプション
     * @param {number} [options.maxSlotsPerWorker=4] - ワーカーごとの最大スロット数
     * @param {number} [options.chunkSize=12] - チャンクサイズ
     */
    constructor(numWorkers, options = {}) {
        this.workers = [];
        this.slots = new Map();
        this.numWorkers = numWorkers;
        this.options = {
            maxSlotsPerWorker: 4,
            chunkSize: 12,
            ...options,
        };
    }

    /**
     * ワーカーを初期化する
     * @param {string} modelPath - モデルファイルパス
     * @returns {Promise<void>}
     */
    async init(modelPath) {
        const response = await fetch(modelPath);
        const modelBuffer = await response.arrayBuffer();

        for (let i = 0; i < this.numWorkers; i++) {
            const w = new Worker("worker-poc.js", { type: "module" });
            const api = Comlink.wrap(w);
            this.workers.push(api);

            const copy = modelBuffer.slice(0);
            await api.init(Comlink.transfer(copy, [copy]), {
                maxSlots: this.options.maxSlotsPerWorker,
                chunkSize: this.options.chunkSize,
            });
        }
    }

    /**
     * スロットを作成する
     * @param {string} slotId - スロットID
     * @param {number} workerIndex - ワーカーインデックス
     * @param {Object} options - スロットオプション
     * @returns {Promise<Object>} スロット情報（UIに依存しない）
     */
    async createSlot(slotId, workerIndex, options = {}) {
        const api = this.workers[workerIndex];
        await api.createSlot(slotId, options);

        const slotInfo = {
            slotId,
            workerIndex,
            api,
        };

        this.slots.set(slotId, slotInfo);
        return slotInfo;
    }

    /**
     * スロットにフレームを配信する
     * @param {string} slotId - スロットID
     * @param {Float32Array[]} frames
     * @returns {Promise<void>}
     */
    async pushFrames(slotId, frames) {
        const slot = this.slots.get(slotId);
        if (!slot) {
            throw new Error(`Slot ${slotId} not found`);
        }
        await this.workers[slot.workerIndex].pushFrames(slotId, frames);
    }

    /**
     * スロット情報を取得する
     * @param {string} slotId
     * @returns {Object|undefined}
     */
    getSlot(slotId) {
        return this.slots.get(slotId);
    }

    /**
     * 全スロットを取得する
     * @returns {Array<Object>}
     */
    getAllSlots() {
        return Array.from(this.slots.values());
    }

    /**
     * パフォーマンス統計を取得する
     * @returns {Promise<Object>}
     */
    async getPerformanceStats() {
        const results = await Promise.all(
            this.workers.map(api => api.performanceStats)
        );
        return {
            totalWorkers: this.numWorkers,
            totalSlots: results.reduce((sum, r) => sum + r.totalSlots, 0),
            processingSlots: results.reduce((sum, r) => sum + r.processingSlots, 0),
            totalInferenceTime: results.reduce((sum, r) => sum + r.totalInferenceTime, 0),
        };
    }
}

let multiStreamProxy = null;
let slotProxies = new Map();
let isStreaming = false;
let streamIntervalId = null;

const logEl = document.getElementById("log");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const container = document.getElementById("worker-container");

/**
 * ログを出力する
 * @param {string} msg
 */
function log(msg) {
    if (logEl) {
        logEl.textContent += msg + "\n";
        logEl.scrollTop = logEl.scrollHeight;
    }
    console.log(`[Main] ${msg}`);
}

/**
 * スロットカードを作成する
 * @param {string} slotId
 * @returns {HTMLElement}
 */
function createSlotCard(slotId) {
    const div = document.createElement("div");
    div.className = "worker-card";
    div.innerHTML = `
        <div class="worker-id">${slotId}</div>
        <div class="worker-status">
            <div class="busy-indicator idle"></div>
            <span class="status-label">Ready</span>
        </div>
        <div class="decoded-text" style="font-family: monospace; background: #eee; padding: 4px; margin-top: 4px; min-height: 1.2em; word-break: break-all;"></div>
    `;
    return div;
}

/**
 * 初期化
 */
async function init() {
    log("Initializing Refined PoC...");
    try {
        multiStreamProxy = new MultiStreamProxy(NUM_WORKERS, {
            maxSlotsPerWorker: SLOTS_PER_WORKER,
            chunkSize: CHUNK_SIZE,
        });
        await multiStreamProxy.init(MODEL_PATH);

        // スロットを作成
        for (let i = 0; i < NUM_WORKERS; i++) {
            for (let j = 0; j < SLOTS_PER_WORKER; j++) {
                const slotId = `w${i}-s${j}`;
                const card = createSlotCard(slotId);
                container.appendChild(card);

                await multiStreamProxy.createSlot(slotId, i);

                // SlotProxyを作成（UIと結合）
                const slotInfo = multiStreamProxy.getSlot(slotId);
                const proxy = new SlotProxy(slotInfo.api, slotId);
                proxy.indicator = card.querySelector(".busy-indicator");
                proxy.labelEl = card.querySelector(".status-label");
                proxy.textEl = card.querySelector(".decoded-text");
                await proxy.connect();

                slotProxies.set(slotId, proxy);
            }
        }

        log("Ready.");
        startBtn.disabled = false;
    } catch (e) {
        log(`Init Error: ${e}`);
    }
}

let tickCounter = 0;

/**
 * モックフレームを生成する
 * @param {number} count
 * @returns {Float32Array[]}
 */
function getMockFrames(count) {
    const frames = [];
    for (let i = 0; i < count; i++) {
        const frame = new Float32Array(N_BINS);
        const isOn = (Math.floor(tickCounter / 6) % 2 === 0);
        if (isOn) frame[10] = 10.0;
        tickCounter++;
        frames.push(frame);
    }
    return frames;
}

/**
 * ストリーミングステップ
 */
async function streamingStep() {
    if (!isStreaming) return;
    const frames = getMockFrames(CHUNK_SIZE);
    await Promise.all(Array.from(slotProxies.entries()).map(async ([slotId, proxy]) => {
        await multiStreamProxy.pushFrames(slotId, frames);
        proxy.handleStatus(true);
        try {
            await proxy.api.pushFrames(slotId, frames);
        } finally {
            proxy.handleStatus(false);
        }
    }));
}

startBtn.addEventListener("click", () => {
    isStreaming = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    log("Started.");
    streamIntervalId = setInterval(streamingStep, INTERVAL_MS);
});

stopBtn.addEventListener("click", () => {
    isStreaming = false;
    clearInterval(streamIntervalId);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    log("Stopped.");
    slotProxies.forEach((_, id) => log(`${id} done.`));
});

init();
