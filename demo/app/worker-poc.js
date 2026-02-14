import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js";
import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";
import { StreamInference } from '../stream-inference.js';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

/**
 * StreamManager - ワーカー内のストリーム管理クラス
 *
 * 責務:
 * - ONNXセッションの管理
 * - StreamInference インスタンスの作成・管理
 * - スロットの作成・破棄
 * - フレームの配信
 * - 結果イベントの通知
 * - スロット情報の提供
 */
class StreamManager {
    constructor() {
        /** @type {ort.InferenceSession|null} */
        this.session = null;

        /** @type {Map<string, {inference: StreamInference, text: string}>} */
        this.slots = new Map();

        /** @type {Object} */
        this.options = {
            maxSlots: 4,
            chunkSize: 12,
            useWebGPU: false,
            historyLength: 800,
        };

        console.log("[StreamManager] Ready");
    }

    /**
     * ONNXセッションを初期化する
     * @param {ArrayBuffer} modelBuffer - モデルバッファ
     * @param {Object} options - 設定オプション
     * @param {number} [options.maxSlots=4] - 最大スロット数
     * @param {number} [options.chunkSize=12] - チャンクサイズ
     * @param {boolean} [options.useWebGPU=false] - WebGPU使用フラグ
     * @param {number} [options.historyLength=800] - 履歴長
     * @returns {Promise<boolean>}
     */
    async init(modelBuffer, options = {}) {
        try {
            this.session = await ort.InferenceSession.create(modelBuffer);
            this.options = { ...this.options, ...options };
            return true;
        } catch (e) {
            console.error("[StreamManager] Init error:", e);
            throw e;
        }
    }

    /**
     * 新しいスロットを作成する
     * @param {string} slotId - スロットID
     * @param {Object} options - スロットオプション
     * @returns {Promise<boolean>}
     */
    async createSlot(slotId, options = {}) {
        if (!this.session) {
            throw new Error('Session not initialized');
        }
        if (this.slots.has(slotId)) {
            throw new Error(`Slot ${slotId} already exists`);
        }

        const inference = new StreamInference(this.session, ort, {
            chunkSize: this.options.chunkSize,
            useWebGPU: this.options.useWebGPU,
            historyLength: this.options.historyLength,
        });

        const slot = {
            inference,
            text: '',
        };

        // テキスト更新を購読
        inference.addEventListener('result', (e) => {
            const newChars = e.detail.newChars;
            if (newChars && newChars.length > 0) {
                slot.text += newChars.join('');
            }
        });

        this.slots.set(slotId, slot);
        return true;
    }

    /**
     * スロットを破棄する
     * @param {string} slotId - スロットID
     * @returns {Promise<boolean>}
     */
    async disposeSlot(slotId) {
        const slot = this.slots.get(slotId);
        if (slot) {
            if (slot.inference) {
                slot.inference.dispose();
            }
            this.slots.delete(slotId);
        }
        return true;
    }

    /**
     * スロットにフレームを配信する
     * @param {string} slotId - スロットID
     * @param {Float32Array[]} frames - スペクトログラムフレーム配列
     * @returns {Promise<boolean>}
     */
    async pushFrames(slotId, frames) {
        const slot = this.slots.get(slotId);
        if (!slot) {
            throw new Error(`Slot ${slotId} not found`);
        }

        for (const frame of frames) {
            slot.inference.pushFrame(frame);
        }
        await slot.inference.waitForProcessing();
        return true;
    }

    /**
     * スロットの結果イベントを購読する
     * @param {string} slotId - スロットID
     * @param {Function} onResult - 結果コールバック
     * @returns {Promise<boolean>}
     */
    async subscribe(slotId, onResult) {
        const slot = this.slots.get(slotId);
        if (!slot) {
            throw new Error(`Slot ${slotId} not found`);
        }

        slot.inference.addEventListener('result', (e) => onResult(e.detail));
        return true;
    }

    /**
     * スロット情報を取得する
     * @param {string} slotId - スロットID
     * @returns {Object|null}
     */
    getSlotInfo(slotId) {
        const slot = this.slots.get(slotId);
        if (!slot) return null;

        return {
            slotId,
            text: slot.text,
            inferenceTime: slot.inference.inferenceTime,
            isProcessing: slot.inference.isProcessing,
            frameCount: slot.inference.frameCount,
        };
    }

    /**
     * 全スロット情報を取得する
     * @returns {Array<Object>}
     */
    getAllSlots() {
        return Array.from(this.slots.entries()).map(([slotId, slot]) => ({
            slotId,
            text: slot.text,
            inferenceTime: slot.inference.inferenceTime,
            isProcessing: slot.inference.isProcessing,
            frameCount: slot.inference.frameCount,
        }));
    }

    /**
     * パフォーマンス統計を取得する
     * @returns {Object}
     */
    get performanceStats() {
        const slots = Array.from(this.slots.values());
        return {
            totalSlots: slots.length,
            processingSlots: slots.filter(s => s.inference.isProcessing).length,
            totalInferenceTime: slots.reduce((sum, s) => sum + s.inference.inferenceTime, 0),
        };
    }

    /**
     * 全スロットをリセットする
     */
    reset() {
        for (const slot of this.slots.values()) {
            if (slot.inference) {
                slot.inference.reset();
            }
        }
    }

    /**
     * 全リソースを解放する
     */
    dispose() {
        for (const slot of this.slots.values()) {
            if (slot.inference) {
                slot.inference.dispose();
            }
        }
        this.slots.clear();
        this.session = null;
    }
}

Comlink.expose(new StreamManager());
