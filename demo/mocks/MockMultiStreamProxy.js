/**
 * MockMultiStreamProxy - MultiStreamProxyのモッククラス
 *
 * テスト用に使用するモッククラス。
 * Workerを使用せず、メモリ内でスロットを管理する。
 */

/**
 * MockSlotApiProxy - スロットごとのAPI操作をモックするクラス
 */
class MockSlotApiProxy {
    constructor(workerApi, slotId) {
        this.workerApi = workerApi;
        this.slotId = slotId;
    }

    async pushFrames(frames) {
        await this.workerApi.pushFrames(this.slotId, frames);
    }

    async getSlotInfo() {
        const slotInfo = await this.workerApi.getSlotInfo(this.slotId);
        return {
            slotId: this.slotId,
            ...slotInfo,
        };
    }

    async getSignalHistory() {
        return [];
    }

    async getEvents() {
        return [];
    }

    async waitForProcessing() {
        return Promise.resolve();
    }

    async reset() {
        await this.workerApi.resetSlot(this.slotId);
    }

    get frameCount() {
        const slot = this.workerApi.slots.get(this.slotId);
        return slot ? slot.frameCount : 0;
    }

    get inferenceTime() {
        const slot = this.workerApi.slots.get(this.slotId);
        return slot ? slot.inferenceTime : 0;
    }

    get isProcessing() {
        const slot = this.workerApi.slots.get(this.slotId);
        return slot ? slot.isProcessing : false;
    }
}

export class MockMultiStreamProxy {
    constructor() {
        this.slots = new Map();
        this.inits = [];
        this.pushes = [];
        this.resets = [];
    }

    async init(modelBuffer, numWorkers, options) {
        this.inits.push({ modelBuffer, numWorkers, options });
        return Promise.resolve();
    }

    async createSlot(slotId, workerIndex, options) {
        this.slots.set(slotId, { slotId, workerIndex, text: '', inferenceTime: 0, isProcessing: false, frameCount: 0 });
        return Promise.resolve();
    }

    async pushFrames(slotId, frames) {
        this.pushes.push({ slotId, frames });
        return Promise.resolve();
    }

    async resetSlot(slotId) {
        this.resets.push(slotId);
        return Promise.resolve();
    }

    async getSlotInfo(slotId) {
        const slot = this.slots.get(slotId);
        if (!slot) return null;
        return {
            slotId,
            text: slot.text,
            inferenceTime: slot.inferenceTime,
            isProcessing: slot.isProcessing,
            frameCount: slot.frameCount,
        };
    }

    async getAllSlots() {
        return Array.from(this.slots.values());
    }

    getSlot(slotId) {
        const slot = this.slots.get(slotId);
        if (!slot) return null;
        return new MockSlotApiProxy(this, slotId);
    }

    // テスト用ヘルパーメソッド
    setSlotInferenceTime(slotId, time) {
        const slot = this.slots.get(slotId);
        if (slot) {
            slot.inferenceTime = time;
        }
    }

    dispose() {
        this.slots.clear();
    }
}
