import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";

/**
 * SlotApiProxy - スロットごとのAPI操作をカプセル化するクラス
 *
 * 責務:
 * - ワーカーAPIとスロットIDを保持
 * - スロットに関連するすべての操作を提供
 *   - フレームの配信
 *   - StreamInferenceのメソッド呼び出し
 *   - スロットの推論状態のリセット
 *   - スロット情報の取得
 */
export class SlotApiProxy {
    /**
     * @param {Object} workerApi - ワーカーAPI（ComlinkでラップされたWorker API）
     * @param {string} slotId - スロットID
     */
    constructor(workerApi, slotId) {
        this.workerApi = workerApi;
        this.slotId = slotId;
    }

    /**
     * スロットにフレームを配信する
     * @param {Float32Array[]} frames
     * @returns {Promise<void>}
     */
    async pushFrames(frames) {
        await this.workerApi.pushFrames(this.slotId, frames);
    }

    /**
     * スロット情報を取得する
     * @returns {Promise<Object>}
     */
    async getSlotInfo() {
        const slotInfo = await this.workerApi.getSlotInfo(this.slotId);
        return {
            slotId: this.slotId,
            ...slotInfo,
        };
    }

    /**
     * シグナル履歴を取得する
     * @returns {Promise<Array<Object>>}
     */
    async getSignalHistory() {
        return await this.workerApi.getSlotSignalHistory(this.slotId);
    }

    /**
     * イベント履歴を取得する
     * @returns {Promise<Array<Object>>}
     */
    async getEvents() {
        return await this.workerApi.getSlotEvents(this.slotId);
    }

    /**
     * 処理完了を待つ
     * @returns {Promise<void>}
     */
    async waitForProcessing() {
        return await this.workerApi.waitForSlotProcessing(this.slotId);
    }

    /**
     * 推論状態をリセットする
     * @returns {Promise<void>}
     */
    async reset() {
        await this.workerApi.resetSlot(this.slotId);
    }

    /**
     * フレーム数を取得する
     * @returns {number}
     */
    get frameCount() {
        return this.workerApi.getSlotFrameCount(this.slotId);
    }

    /**
     * 推論時間を取得する
     * @returns {number}
     */
    get inferenceTime() {
        return this.workerApi.getSlotInferenceTime(this.slotId);
    }

    /**
     * 処理中かどうかを取得する
     * @returns {boolean}
     */
    get isProcessing() {
        return this.workerApi.getSlotIsProcessing(this.slotId);
    }
}

/**
 * MultiStreamProxy - 複数ワーカーを管理するクラス
 *
 * 責務:
 * - 複数のワーカーを初期化・管理する
 * - ワーカーごとのスロットを作成・管理する
 * - スロットごとのSlotApiProxyを提供する
 * - パフォーマンス統計を収集する
 */
export class MultiStreamProxy {
    /**
     * @param {number} numWorkers - ワーカー数
     */
    constructor(numWorkers) {
        this.workers = [];
        this.workerInstances = [];
        this.slots = new Map();
        this.numWorkers = numWorkers;
    }

    /**
     * ワーカーを初期化する
     * @param {ArrayBuffer} modelBuffer - モデルバッファ
     * @param {number} numWorkers - ワーカー数
     * @param {Object} options - 設定オプション
     * @param {string} [options.workerURL="./worker-multi-stream.js"] - ワーカーファイルのパス
     * @param {number} [options.maxSlotsPerWorker=4] - ワーカーごとの最大スロット数
     * @param {number} [options.chunkSize=12] - チャンクサイズ
     * @param {boolean} [options.useWebGPU=false] - WebGPUを使用するか
     * @param {number} [options.historyLength=800] - ヒストリ長
     * @returns {Promise<void>}
     */
    async init(modelBuffer, numWorkers, options = {}) {
        this.numWorkers = numWorkers;
        const {
            workerURL = "/worker-multi-stream.js",
            maxSlotsPerWorker = 4,
            chunkSize = 12,
            useWebGPU = false,
            historyLength = 800,
        } = options;

        for (let i = 0; i < this.numWorkers; i++) {
            try {
                console.log(`[MultiStreamProxy] Initializing worker ${i}...`);
                const w = new Worker(workerURL, { type: "module" });
                const api = Comlink.wrap(w);
                this.workers.push(api);
                this.workerInstances.push(w);

                const copy = modelBuffer.slice(0);
                console.log(`[MultiStreamProxy] Calling init on worker ${i}...`);
                await api.init(Comlink.transfer(copy, [copy]), {
                    maxSlots: maxSlotsPerWorker,
                    chunkSize: chunkSize,
                    useWebGPU: useWebGPU,
                    historyLength: historyLength,
                });
                console.log(`[MultiStreamProxy] Worker ${i} initialized successfully`);
            } catch (e) {
                console.error(`[MultiStreamProxy] Error initializing worker ${i}:`, e);
                throw e;
            }
        }

        // スロットもoptionsに設定した分初期化
        const totalSlots = this.numWorkers * maxSlotsPerWorker;
        console.log(`[MultiStreamProxy] Creating ${totalSlots} slots...`);
        for (let i = 0; i < totalSlots; i++) {
            const workerIndex = Math.floor(i / maxSlotsPerWorker);
            const slotId = `worker-${workerIndex}-slot-${i}`;
            await this.createSlot(slotId, workerIndex, {});
        }
        console.log(`[MultiStreamProxy] All slots created`);
    }

    /**
     * スロットを作成する
     * @param {string} slotId - スロットID
     * @param {number} workerIndex - ワーカーインデックス
     * @param {Object} options - スロットオプション
     * @returns {Promise<SlotApiProxy>} SlotApiProxyインスタンス
     */
    async createSlot(slotId, workerIndex, options = {}) {
        const api = this.workers[workerIndex];
        await api.createSlot(slotId, options);

        const slotProxy = new SlotApiProxy(api, slotId);
        this.slots.set(slotId, slotProxy);
        return slotProxy;
    }

    /**
     * スロットのSlotApiProxyを取得する
     * @param {string} slotId
     * @returns {SlotApiProxy|null}
     */
    getSlot(slotId) {
        const slotProxy = this.slots.get(slotId);
        if (!slotProxy) return null;
        return slotProxy;
    }

    /**
     * 全スロットを取得する
     * @returns {Promise<Array<Object>>}
     */
    async getAllSlots() {
        const results = await Promise.all(
            this.workers.map(api => api.getAllSlots())
        );
        const allSlots = [];
        for (const workerSlots of results) {
            allSlots.push(...workerSlots);
        }
        return allSlots;
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

    /**
     * 全リソースを解放する
     */
    dispose() {
        for (const api of this.workers) {
            api.dispose();
        }
        for (const w of this.workerInstances) {
            w.terminate();
        }
        this.slots.clear();
        this.workers = [];
        this.workerInstances = [];
    }
}
