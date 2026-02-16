/**
 * MultiStreamManager - 複数ピークの同時デコード管理
 *
 * 責務:
 * - スロットIDへの周波数の割り当て
 * - ピークとスロットとのマッピング
 * - スロットル管理
 * - メイン/サブ周波数の区別管理
 *
 * 設計方針:
 * - Single Source of Truth: スロットの状態（周波数、割り当て状況）は Manager 内の _slots (Map<string, SlotState>) で管理する。
 *   ワーカー側の状態には依存せず、Manager が一方的にコマンドを送る。
 * - Race Condition 対策: AsyncLock を使用し、updatePeaks などの非同期処理が重複しないように制御する。
 *   処理中に新しいリクエストが来た場合は、待たずにスキップする（最新の処理のみを行えばよいため）。
 */

import { StreamInference, N_BINS } from './stream-inference.js';

// --- Constants ---
const NEARBY_HZ = 50;           // ピーク追従の近傍判定 (±Hz)
const SLOT_TIMEOUT_MS = 10000;  // ピーク消失後にスロット解放するまでの時間
const SLOT_COOLDOWN_MS = 20000; // スロット作成後、SNR置換を禁止する時間
const SNR_REPLACE_MARGIN = 2.0; // SNR置換に必要な倍率（ヒステリシス）
const EMA_ALPHA = 0.1;          // 推論時間のEMA平滑化係数
const THROTTLE_THRESHOLD = 0.95; // 予算の95%超過でスロットル
const RECOVER_THRESHOLD = 0.7;  // 予算の70%未満で復帰

/**
 * 簡易 AsyncLock (Mutex)
 * 処理中の再入を防ぎ、スキップするためのロック
 */
class AsyncLock {
    constructor() {
        this._locked = false;
    }

    /**
     * ロックを取得して関数を実行する。
     * すでにロックされている場合は実行せずに false を返す。
     * @param {Function} task - 非同期タスク
     * @returns {Promise<boolean>} タスクが実行されたかどうか
     */
    async tryLock(task) {
        if (this._locked) {
            return false;
        }
        this._locked = true;
        try {
            await task();
            return true;
        } finally {
            this._locked = false;
        }
    }
}

/**
 * スロットの状態を管理するクラス
 */
class SlotState {
    /**
     * @param {string} id - スロットID
     */
    constructor(id) {
        this.id = id;
        this.freq = 0;          // 割り当て周波数 (0 = 未割り当て)
        this.snr = 0;           // 最新SNR
        this.text = '';         // 最新デコードテキスト
        this.lastSeen = 0;      // 最後にピークが検出された時間
        this.lastTextUpdate = 0;// 最後にテキストが更新された時間
        this.throttled = false; // スロットル状態
        this.avgInferenceTime = 0; // 推論時間のEMA
    }

    reset() {
        this.freq = 0;
        this.snr = 0;
        this.text = '';
        this.lastSeen = 0;
        this.lastTextUpdate = 0;
        this.throttled = false;
        this.avgInferenceTime = 0;
        // id は維持
    }
}

/**
 * スペクトル magnitudes から指定中心周波数を基準に N_BINS 幅の
 * スペクトログラムフレームを線形補間で抽出する。
 *
 * @param {Float32Array} magnitudes - ワイドバンドFFTパワースペクトル
 * @param {number} centerFreq - 中心周波数 (Hz)
 * @param {number} nFft - FFTサイズ
 * @param {number} sampleRate - サンプルレート
 * @returns {Float32Array} [N_BINS] のスペクトログラムフレーム
 */
export function extractSpecFrame(magnitudes, centerFreq, nFft, sampleRate) {
    const binBW = sampleRate / nFft;
    const W = N_BINS * binBW;
    const fStart = centerFreq - W / 2 + binBW / 2;
    const specFrame = new Float32Array(N_BINS);

    for (let i = 0; i < N_BINS; i++) {
        const f = fStart + i * binBW;
        const k = f * nFft / sampleRate;
        const kIdx = Math.floor(k);
        const kFrac = k - kIdx;

        if (kIdx >= 0 && kIdx < magnitudes.length - 1) {
            specFrame[i] = magnitudes[kIdx] * (1 - kFrac) + magnitudes[kIdx + 1] * kFrac;
        }
    }

    return specFrame;
}

/**
 * PerformanceStats
 * @typedef {Object} PerformanceStats
 * @property {number} totalInferenceTime
 * @property {number} budget
 * @property {number} utilization
 * @property {number} activeSlots
 * @property {number} maxSlots
 * @property {boolean} throttled
 */

export class MultiStreamManager {
    /**
     * @param {ArrayBuffer} modelBuffer - ONNXモデルバッファ
     * @param {Object} options
     * @param {Object} options.multiStreamProxy - MultiStreamProxyインスタンス（必須）
     * @param {number} [options.numWorkers=4] - ワーカー数
     * @param {number} [options.maxSlots=4] - 最大同時デコード数
     * @param {number} [options.chunkSize=12] - StreamInference の chunkSize
     * @param {number} [options.hopMs=10] - フレーム間隔(ms)
     * @param {number} [options.historyLength=800] - 可視化用ヒストリ長
     * @param {boolean} [options.useWebGPU=false]
     * @param {string} [options.workerURL="./worker-multi-stream.js"] - ワーカーファイルのパス
     */
    constructor(modelBuffer, options = {}) {
        if (!modelBuffer) throw new Error('MultiStreamManager requires modelBuffer');
        if (!options.multiStreamProxy) throw new Error('MultiStreamManager requires multiStreamProxy');

        this._modelBuffer = modelBuffer;
        this._numWorkers = options.numWorkers || 2;
        this._maxSlots = options.maxSlots || 8;
        this._chunkSize = options.chunkSize || 12;
        this._hopMs = options.hopMs || 10;
        this._historyLength = options.historyLength || 800;
        this._useWebGPU = options.useWebGPU || false;
        this._workerURL = options.workerURL;

        // 時間管理（テスト用に注入可能）
        this._now = options.now || Date.now;

        // 時間予算 = chunkSize * hopMs
        this._budgetMs = this._chunkSize * this._hopMs;

        // スロット管理 (Source of Truth)
        /** @type {Map<string, SlotState>} */
        this._slots = new Map();

        // メイン周波数（追従中の周波数）
        this._mainFreq = 700;

        this._disposed = false;
        this._initialized = false;
        this._lock = new AsyncLock();

        // MultiStreamProxyを呼び出し元から受け取る（必須）
        this._multiStreamProxy = options.multiStreamProxy;
    }

    /**
     * 初期化する
     * @returns {Promise<void>}
     */
    async init() {
        if (this._initialized) return;

        console.log("[MultiStreamManager] Starting init...");
        console.log("[MultiStreamManager] Initializing MultiStreamProxy...");

        // Proxyの初期化（ワーカー起動）
        await this._multiStreamProxy.init(this._modelBuffer, this._numWorkers, {
            workerURL: this._workerURL,
            maxSlotsPerWorker: Math.ceil(this._maxSlots / this._numWorkers), // 十分な数を確保
            chunkSize: this._chunkSize,
            useWebGPU: this._useWebGPU,
            historyLength: this._historyLength,
        });

        // スロット情報の初期化 (Proxyが作成したスロットIDを取得して管理下に置く)
        const proxySlots = await this._multiStreamProxy.getAllSlots();

        // すべてのスロットをサブスロットとして管理
        for (const s of proxySlots) {
            const slotState = new SlotState(s.slotId);
            this._slots.set(s.slotId, slotState);
        }

        console.log(`[MultiStreamManager] Initialized with ${this._slots.size} slots.`);
        this._initialized = true;
    }

    /**
     * スロットの状態を取得する (内部使用)
     * @param {string} slotId
     * @returns {SlotState}
     */
    _getSlot(slotId) {
        return this._slots.get(slotId);
    }

    /**
     * スロットIDから周波数を取得する（互換用）
     * @param {string} slotId
     */
    _getFreqForSlotId(slotId) {
        const slot = this._slots.get(slotId);
        return slot ? slot.freq : 0;
    }

    /**
     * ピーク検出結果を受け取り、スロット割り当てを更新する。
     * AsyncLock により、実行中の場合はスキップされる。
     *
     * @param {Array<{f: number, snr: number}>} peaks - ピーク検出結果 (SNR降順)
     * @param {number} mainFreq - メイン追従中の中心周波数
     * @returns {Promise<void>}
     */
    async updatePeaks(peaks, mainFreq) {
        if (this._disposed) return;

        // ロックが取れない（処理中）ならスキップ（最新のピーク情報だけで処理すれば十分なため）
        const executed = await this._lock.tryLock(async () => {
            await this._updatePeaksInternal(peaks, mainFreq);
        });

        if (!executed) {
            // console.debug("[MultiStreamManager] updatePeaks skipped due to lock.");
        }
    }

    /**
     * ピーク更新の内部ロジック (Locked)
     * @private
     */
    async _updatePeaksInternal(peaks, mainFreq) {
        this._mainFreq = mainFreq;
        const now = this._now();

        // ワーカーからの最新テキスト・統計を取得してローカル状態を更新
        const workerSlots = await this._multiStreamProxy.getAllSlots();
        for (const ws of workerSlots) {
            const slot = this._slots.get(ws.slotId);
            if (slot) {
                if (ws.text && ws.text.length > slot.text.length) {
                    slot.lastTextUpdate = now;
                }
                slot.text = ws.text;
                slot.avgInferenceTime = slot.avgInferenceTime * (1 - EMA_ALPHA) + (ws.inferenceTime || 0) * EMA_ALPHA;
            }
        }

        // ピークとサブスロットのマッチング
        const matchedPeaks = new Set();

        // 既存サブスロットへの近傍マッチング
        for (const slot of this._slots.values()) {
            if (slot.freq === 0) continue; // 未割り当て

            // メイン周波数に吸われた場合は解放（メインが優先）
            if (Math.abs(slot.freq - mainFreq) < NEARBY_HZ) {
                // console.log(`[MSM] Slot ${slot.id} (${Math.round(slot.freq)}Hz) absorbed by main freq.`);
                await this._releaseSlot(slot);
                continue;
            }

            // 最も近いピークを探す
            let closestPeak = null;
            let minDiff = Infinity;
            for (let i = 0; i < peaks.length; i++) {
                if (matchedPeaks.has(i)) continue;
                const diff = Math.abs(peaks[i].f - slot.freq);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestPeak = { peak: peaks[i], idx: i };
                }
            }

            if (closestPeak && minDiff < NEARBY_HZ) {
                // 近傍: マッチ
                matchedPeaks.add(closestPeak.idx);
                // 周波数更新 (Smoothing)
                slot.freq = slot.freq * 0.9 + closestPeak.peak.f * 0.1;
                slot.snr = closestPeak.peak.snr;
                slot.lastSeen = now;
            }
        }

        // 新規割り当てと置換
        for (let i = 0; i < peaks.length; i++) {
            if (matchedPeaks.has(i)) continue;
            const peak = peaks[i];

            // メイン周波数に近いピークはスキップ（メインスロットで扱う）
            if (Math.abs(peak.f - mainFreq) < NEARBY_HZ) continue;

            // 既存スロットに近いピークもスキップ
            let tooClose = false;
            for (const slot of this._slots.values()) {
                if (slot.freq > 0 && Math.abs(slot.freq - peak.f) < NEARBY_HZ) {
                    tooClose = true;
                    break;
                }
            }
            if (tooClose) continue;

            // 空きスロットを探す
            const freeSlot = Array.from(this._slots.values()).find(s => s.freq === 0);

            if (freeSlot) {
                // 新規割り当て
                await this._assignSlot(freeSlot, peak.f, peak.snr, now);
            } else {
                // 空きなし: 置換判定
                // スロット保有数チェック (maxSlots)
                let activeCount = 0;
                for (const s of this._slots.values()) { if (s.freq > 0) activeCount++; }

                if (activeCount >= this._maxSlots) {
                    // 最弱スロットを探す
                    let worstSlot = null;
                    let worstSnr = Infinity;

                    for (const slot of this._slots.values()) {
                        if (slot.freq === 0) continue; // 未割り当てはスキップ

                        // クールダウン中は保護 (最後にテキストが出てから一定時間)
                        if ((now - slot.lastTextUpdate) < SLOT_COOLDOWN_MS) continue;

                        if (slot.snr < worstSnr) {
                            worstSnr = slot.snr;
                            worstSlot = slot;
                        }
                    }

                    if (worstSlot && peak.snr > worstSnr * SNR_REPLACE_MARGIN) {
                        console.log(`[MSM] Converting slot ${worstSlot.id} (${Math.round(worstSlot.freq)}Hz -> ${Math.round(peak.f)}Hz)`);
                        await this._assignSlot(worstSlot, peak.f, peak.snr, now);
                    }
                } else {
                    // 論理的にはまだ割り当てられるはずだが、物理スロットが足りない（バグ？）
                    // initで十分確保しているはずなので、ここには来ないはず。
                }
            }
        }

        // タイムアウト解放
        for (const slot of this._slots.values()) {
            if (slot.freq === 0) continue;

            // クールダウン中は保護
            if ((now - slot.lastTextUpdate) < SLOT_COOLDOWN_MS) continue;

            if ((now - slot.lastSeen) > SLOT_TIMEOUT_MS) {
                console.log(`[MSM] Slot ${slot.id} timed out. (${Math.round(slot.freq)}Hz)`);
                await this._releaseSlot(slot);
            }
        }

        // スロットル制御
        await this._updateThrottling();
    }

    /**
     * スロットに周波数を割り当てる
     * @private
     */
    async _assignSlot(slot, freq, snr, now) {
        slot.freq = freq;
        slot.snr = snr;
        slot.lastSeen = now;
        // リセット
        slot.text = '';
        slot.lastTextUpdate = 0; // 新規なので0
        // Workerの状態リセット
        await this._multiStreamProxy.getSlot(slot.id).reset();
    }

    /**
     * スロットを解放する
     * @private
     */
    async _releaseSlot(slot) {
        slot.reset(); // freq=0 になる
        // Workerの状態リセット
        const proxy = this._multiStreamProxy.getSlot(slot.id);
        if (proxy) await proxy.reset();
    }

    /**
     * パフォーマンス制御
     * @private
     */
    async _updateThrottling() {
        const stats = await this.performanceStats();

        if (stats.utilization > THROTTLE_THRESHOLD) {
            // スロットル対象：SNRが低く、まだスロットルされていないもの
            let target = null;
            let minSnr = Infinity;
            for (const slot of this._slots.values()) {
                if (slot.freq === 0) continue;
                if (!slot.throttled && slot.snr < minSnr) {
                    minSnr = slot.snr;
                    target = slot;
                }
            }
            if (target) {
                target.throttled = true;
                // console.log(`[MSM] Throttled slot ${target.id}`);
            }
        } else if (stats.utilization < RECOVER_THRESHOLD) {
            // 復帰：スロットルされているもののうち、SNRが高いもの（あるいは適当に）
            const throttledSlot = Array.from(this._slots.values()).find(s => s.throttled);
            if (throttledSlot) {
                throttledSlot.throttled = false;
                // console.log(`[MSM] Unthrottled slot ${throttledSlot.id}`);
                // 復帰時はモデルの状態が飛んでいるのでリセット推奨
                const proxy = this._multiStreamProxy.getSlot(throttledSlot.id);
                if (proxy) await proxy.reset();
            }
        }
    }

    /**
     * 全スロットにスペクトルフレームを配信する。
     * @param {Float32Array} magnitudes 
     * @param {number} nFft 
     * @param {number} sampleRate 
     */
    async pushMagnitudes(magnitudes, nFft, sampleRate) {
        if (this._disposed) return;

        // 以下のループは非同期ロックを行わない（パフォーマンス重視）
        // メインスレッドでの実行なので、SlotStateの参照競合は起きない（JSはシングルスレッド）
        // ただし updatePeaks の await 中にここが割り込む可能性はあるが、
        // freq などの値はアトミックに読めるので問題ない。

        const tasks = [];
        for (const slot of this._slots.values()) {
            if (slot.throttled) continue;
            if (slot.freq === 0) continue;

            const proxy = this._multiStreamProxy.getSlot(slot.id);
            if (!proxy) continue;

            const specFrame = extractSpecFrame(magnitudes, slot.freq, nFft, sampleRate);

            // Promise.all で待たない方が良いかもしれないが、
            // ここでは await せず fire and forget にする
            tasks.push(proxy.pushFrames([specFrame]).catch(e => console.error(e)));
        }
        await Promise.all(tasks);
    }

    /**
     * サブスロット情報取得 (UI用)
     */
    async getSubSlots() {
        const result = [];
        for (const slot of this._slots.values()) {
            if (slot.freq === 0) continue;
            result.push({
                freq: slot.freq,
                text: slot.text, // Workerから同期された最新テキスト
                snr: slot.snr,
                throttled: slot.throttled
            });
        }
        return result;
    }

    /**
     * 全スロット情報取得
     */
    async getAllSlots() {
        const result = [];
        for (const slot of this._slots.values()) {
            result.push({
                slotId: slot.id,
                freq: slot.freq,
                text: slot.text,
                snr: slot.snr,
                throttled: slot.throttled,
                avgInferenceTime: slot.avgInferenceTime
            });
        }
        return result;
    }

    /**
     * 指定周波数のテキスト取得
     */
    async getTextForFreq(freq, tolerance = NEARBY_HZ) {
        for (const slot of this._slots.values()) {
            if (slot.freq > 0 && Math.abs(slot.freq - freq) < tolerance) {
                return slot.text;
            }
        }
        return null;
    }

    /**
     * 最大スロット数変更
     */
    async setMaxSlots(maxSlots) {
        this._maxSlots = maxSlots;
        console.warn("[MultiStreamManager] setMaxSlots is partially supported (logical limit only).");

        // 論理的な制限を超えている分を解放する
        let activeCount = 0;
        const activeSubs = [];
        for (const slot of this._slots.values()) {
            if (slot.freq > 0) {
                activeCount++;
                activeSubs.push(slot);
            }
        }

        if (activeCount > this._maxSlots) {
            // SNR昇順（弱い順）にソート
            activeSubs.sort((a, b) => a.snr - b.snr);

            // 超過分
            const removeCount = activeCount - this._maxSlots;
            for (let i = 0; i < removeCount; i++) {
                await this._releaseSlot(activeSubs[i]);
            }
        }
    }

    /**
     * パフォーマンス統計
     */
    async performanceStats() {
        let totalTime = 0;
        let activeCount = 0;
        let anyThrottled = false;

        for (const slot of this._slots.values()) {
            if (slot.throttled) {
                anyThrottled = true;
                continue;
            }
            if (slot.freq === 0) continue;

            totalTime += slot.avgInferenceTime;
            activeCount++;
        }

        return {
            totalInferenceTime: totalTime,
            budget: this._budgetMs,
            utilization: this._budgetMs > 0 ? totalTime / this._budgetMs : 0,
            activeSlots: activeCount,
            maxSlots: this._maxSlots,
            throttled: anyThrottled,
        };
    }

    async reset() {
        for (const slot of this._slots.values()) {
            slot.reset();

            const proxy = this._multiStreamProxy.getSlot(slot.id);
            if (proxy) await proxy.reset();
        }
    }

    dispose() {
        this._disposed = true;
        this._multiStreamProxy.dispose();
        this._slots.clear();
    }

    // Test helper
    _updateSlotFreq(slotId, freq) {
        const slot = this._slots.get(slotId);
        if (slot) slot.freq = freq;
    }
}
