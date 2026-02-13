/**
 * MultiStreamManager - 複数ピークの同時デコード管理
 *
 * 責務:
 * - StreamInference インスタンスの動的割り当て・解放
 * - ピーク検出結果から各スロットへのフレーム配信
 * - メイン/サブ周波数の区別管理
 * - 推論パフォーマンスの測定と動的スロット削減
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
 * @typedef {Object} Slot
 * @property {number} freq - 中心周波数
 * @property {StreamInference} inference - StreamInference インスタンス
 * @property {string} text - デコード済みテキスト
 * @property {number} snr - 最新SNR (linear)
 * @property {number} lastSeen - 最後にピーク検出されたタイムスタンプ
 * @property {boolean} isMain - メイン周波数かどうか
 * @property {boolean} throttled - パフォーマンス制限で一時停止中か
 * @property {number} avgInferenceTime - EMA平滑化された推論時間(ms)
 */

/**
 * @typedef {Object} PerformanceStats
 * @property {number} totalInferenceTime - 全アクティブスロットの推論時間合計(ms)
 * @property {number} budget - 時間予算(ms)
 * @property {number} utilization - 予算利用率 (0.0-1.0+)
 * @property {number} activeSlots - アクティブスロット数
 * @property {number} maxSlots - 最大スロット数
 * @property {boolean} throttled - いずれかのスロットがスロットル中か
 */

export class MultiStreamManager {
    /**
     * @param {ort.InferenceSession} session - ONNX Runtime inference session (全スロットで共有)
     * @param {typeof ort} ort - ONNX Runtime module
     * @param {Object} options
     * @param {number} [options.maxSlots=4] - 最大同時デコード数
     * @param {number} [options.chunkSize=12] - StreamInference の chunkSize
     * @param {number} [options.hopMs=10] - フレーム間隔(ms)
     * @param {number} [options.historyLength=800] - 可視化用ヒストリ長
     * @param {boolean} [options.useWebGPU=false]
     */
    constructor(session, ort, options = {}) {
        if (!session) throw new Error('MultiStreamManager requires an ONNX session');
        if (!ort) throw new Error('MultiStreamManager requires ort module');

        this._session = session;
        this._ort = ort;
        this._maxSlots = options.maxSlots || 4;
        this._chunkSize = options.chunkSize || 12;
        this._hopMs = options.hopMs || 10;
        this._historyLength = options.historyLength || 800;
        this._useWebGPU = options.useWebGPU || false;

        // 時間予算 = chunkSize * hopMs
        this._budgetMs = this._chunkSize * this._hopMs;

        /** @type {Slot[]} */
        this._slots = [];

        this._disposed = false;
    }

    /**
     * 新しいスロットを作成する
     * @param {number} freq
     * @param {number} snr
     * @param {boolean} isMain
     * @returns {Slot}
     * @private
     */
    _createSlot(freq, snr, isMain) {
        const inference = new StreamInference(this._session, this._ort, {
            chunkSize: this._chunkSize,
            useWebGPU: this._useWebGPU,
            historyLength: this._historyLength,
        });

        const slot = {
            freq,
            inference,
            text: '',
            snr,
            lastSeen: Date.now(),
            lastTextUpdate: 0,
            isMain,
            throttled: false,
            avgInferenceTime: 0,
        };

        // テキスト更新を購読（newChars を追記、inference.reset() の影響を受けない）
        inference.addEventListener('result', (e) => {
            const newChars = e.detail.newChars;
            if (newChars && newChars.length > 0) {
                slot.text += newChars.join('');
                slot.lastTextUpdate = Date.now();
            }
        });

        console.log(`[MSM] スロット割当: ${isMain ? 'MAIN' : 'SUB'} freq=${Math.round(freq)}Hz snr=${snr.toFixed(1)}`);
        return slot;
    }

    /**
     * スロットを破棄する
     * @param {Slot} slot
     * @private
     */
    _destroySlot(slot) {
        console.log(`[MSM] スロット解放: ${slot.isMain ? 'MAIN' : 'SUB'} freq=${Math.round(slot.freq)}Hz text="${slot.text.slice(-30)}"`);
        if (slot.inference) {
            slot.inference.dispose();
            slot.inference = null;
        }
    }

    /**
     * ピーク検出結果を受け取り、スロット割り当てを更新する。
     *
     * @param {Array<{f: number, snr: number}>} peaks - ピーク検出結果 (SNR降順)
     * @param {number} mainFreq - メイン追従中の中心周波数
     */
    updatePeaks(peaks, mainFreq) {
        if (this._disposed) return;

        const now = Date.now();
        const matched = new Set();

        // 1. 既存スロットの近傍マッチング
        for (const slot of this._slots) {
            const nearbyPeak = peaks.find(
                (p, idx) => !matched.has(idx) && Math.abs(p.f - slot.freq) < NEARBY_HZ
            );
            if (nearbyPeak) {
                const peakIdx = peaks.indexOf(nearbyPeak);
                matched.add(peakIdx);
                // スムーズに追従
                slot.freq = slot.freq * 0.9 + nearbyPeak.f * 0.1;
                slot.snr = nearbyPeak.snr;
                slot.lastSeen = now;
            } else if (!slot.isMain) {
                const age = now - slot.lastSeen;
                if (age > 1000 && age % 2000 < 100) { // 頻度を抑えてログ
                    console.log(`[MSM] ピーク未検出: freq=${Math.round(slot.freq)}Hz age=${(age / 1000).toFixed(1)}s/${SLOT_TIMEOUT_MS / 1000}s`);
                }
            }
        }

        // 2. メインスロットの確認・作成
        const mainSlot = this._slots.find(s => s.isMain);
        if (!mainSlot) {
            // メインスロットがない → 作成
            const slot = this._createSlot(mainFreq, 0, true);
            this._slots.unshift(slot); // メインは先頭
        } else {
            // メインスロットはメイン周波数に追従
            mainSlot.freq = mainFreq;
            mainSlot.isMain = true;
        }

        // 3. 新規ピークの割り当て（空きスロットまたはSNR置換）
        const effectiveMax = this._getEffectiveMaxSlots();
        for (let i = 0; i < peaks.length; i++) {
            if (matched.has(i)) continue;
            const peak = peaks[i];

            // メイン周波数と近すぎるピークはスキップ
            if (Math.abs(peak.f - mainFreq) < NEARBY_HZ) continue;

            // 既存スロットと近すぎるピークはスキップ
            const tooClose = this._slots.some(s => Math.abs(s.freq - peak.f) < NEARBY_HZ);
            if (tooClose) continue;

            if (this._slots.length < effectiveMax) {
                // 空きスロットに割り当て
                const slot = this._createSlot(peak.f, peak.snr, false);
                this._slots.push(slot);
            } else {
                // 満杯 → SNRが十分高く、クールダウン済みのサブスロットより強ければ置換
                let worstIdx = -1;
                let worstSnr = Infinity;
                for (let j = 0; j < this._slots.length; j++) {
                    const s = this._slots[j];
                    // 最終テキスト更新からクールダウン経過済みのみ置換対象
                    if (!s.isMain && s.snr < worstSnr && (now - s.lastTextUpdate) > SLOT_COOLDOWN_MS) {
                        worstSnr = s.snr;
                        worstIdx = j;
                    }
                }
                if (worstIdx >= 0 && peak.snr > worstSnr * SNR_REPLACE_MARGIN) {
                    console.log(`[MSM] SNR置換: ${Math.round(this._slots[worstIdx].freq)}Hz(snr=${worstSnr.toFixed(1)}) → ${Math.round(peak.f)}Hz(snr=${peak.snr.toFixed(1)}) margin=${(peak.snr / worstSnr).toFixed(1)}x`);
                    this._destroySlot(this._slots[worstIdx]);
                    this._slots.splice(worstIdx, 1);
                    const slot = this._createSlot(peak.f, peak.snr, false);
                    this._slots.push(slot);
                }
            }
        }

        // 5. タイムアウトしたサブスロットの解放
        //    ただし最近テキストを受信したスロットは保護する
        for (let i = this._slots.length - 1; i >= 0; i--) {
            const slot = this._slots[i];
            if (!slot.isMain && (now - slot.lastSeen) > SLOT_TIMEOUT_MS
                && (now - slot.lastTextUpdate) > SLOT_COOLDOWN_MS) {
                console.log(`[MSM] タイムアウト解放: freq=${Math.round(slot.freq)}Hz age=${((now - slot.lastSeen) / 1000).toFixed(1)}s`);
                this._destroySlot(slot);
                this._slots.splice(i, 1);
            }
        }

        // 6. パフォーマンスベースのスロットル管理
        this._updateThrottling();
    }

    /**
     * 現在のパフォーマンス状態に基づいて有効な最大スロット数を返す。
     * スロットルで削減された数を反映する。
     * @returns {number}
     * @private
     */
    _getEffectiveMaxSlots() {
        return this._maxSlots;
    }

    /**
     * パフォーマンスベースのスロットル管理
     * @private
     */
    _updateThrottling() {
        const stats = this.performanceStats;
        if (stats.utilization > THROTTLE_THRESHOLD) {
            // 予算超過: SNR最低の非スロットル・非メインスロットをスロットル
            let worstSlot = null;
            let worstSnr = Infinity;
            for (const slot of this._slots) {
                if (!slot.isMain && !slot.throttled && slot.snr < worstSnr) {
                    worstSnr = slot.snr;
                    worstSlot = slot;
                }
            }
            if (worstSlot) {
                worstSlot.throttled = true;
            }
        } else if (stats.utilization < RECOVER_THRESHOLD) {
            // 予算に余裕: スロットルされたスロットを1つ復帰
            const throttledSlot = this._slots.find(s => s.throttled);
            if (throttledSlot) {
                throttledSlot.throttled = false;
                // 復帰したスロットの推論状態をリセット（古い状態は使えない）
                // テキストは保持する（newChars 追記方式なので reset しても消えない）
                if (throttledSlot.inference) {
                    throttledSlot.inference.reset();
                }
            }
        }
    }

    /**
     * 全アクティブスロットにスペクトルフレームを配信する。
     *
     * @param {Float32Array} magnitudes - ワイドバンドFFTパワースペクトル
     * @param {number} nFft - FFTサイズ
     * @param {number} sampleRate - サンプルレート
     */
    pushMagnitudes(magnitudes, nFft, sampleRate) {
        if (this._disposed) return;

        for (const slot of this._slots) {
            if (slot.throttled || !slot.inference) continue;

            const specFrame = extractSpecFrame(magnitudes, slot.freq, nFft, sampleRate);
            slot.inference.pushFrame(specFrame);

            // EMA で推論時間を更新
            const lastTime = slot.inference.inferenceTime;
            if (lastTime > 0) {
                slot.avgInferenceTime = slot.avgInferenceTime === 0
                    ? lastTime
                    : EMA_ALPHA * lastTime + (1 - EMA_ALPHA) * slot.avgInferenceTime;
            }
        }
    }

    /**
     * メインスロットの StreamInference を取得する。
     * @returns {StreamInference|null}
     */
    get mainInference() {
        const main = this._slots.find(s => s.isMain);
        return main ? main.inference : null;
    }

    /**
     * メインスロットのデコード済みテキストを取得する。
     * @returns {string}
     */
    get mainText() {
        const main = this._slots.find(s => s.isMain);
        return main ? main.text : '';
    }

    /**
     * サブスロットの情報を取得する（表示用）。
     * @returns {Array<{freq: number, text: string, snr: number, throttled: boolean}>}
     */
    getSubSlots() {
        return this._slots
            .filter(s => !s.isMain)
            .map(s => ({
                freq: s.freq,
                text: s.text,
                snr: s.snr,
                throttled: s.throttled,
            }));
    }

    /**
     * 全スロット情報を取得する（デバッグ/テスト用）。
     * @returns {Array<{freq: number, text: string, snr: number, isMain: boolean, throttled: boolean, avgInferenceTime: number}>}
     */
    getAllSlots() {
        return this._slots.map(s => ({
            freq: s.freq,
            text: s.text,
            snr: s.snr,
            isMain: s.isMain,
            throttled: s.throttled,
            avgInferenceTime: s.avgInferenceTime,
        }));
    }

    /**
     * 指定周波数に最も近いスロットのデコードテキストを取得する。
     * @param {number} freq
     * @param {number} [tolerance=50] - 許容誤差(Hz)
     * @returns {string|null}
     */
    getTextForFreq(freq, tolerance = NEARBY_HZ) {
        const slot = this._slots.find(s => !s.isMain && Math.abs(s.freq - freq) < tolerance);
        return slot ? slot.text : null;
    }

    /**
     * メインスロットの中心周波数を更新する（trackedFreq 変更時）。
     * @param {number} freq
     */
    setMainFreq(freq) {
        const main = this._slots.find(s => s.isMain);
        if (main) {
            main.freq = freq;
        }
    }

    /**
     * 最大スロット数を変更する。減った場合は超過分を解放する。
     * @param {number} maxSlots
     */
    setMaxSlots(maxSlots) {
        this._maxSlots = maxSlots;
        // 超過分のサブスロットを解放
        while (this._slots.length > this._maxSlots) {
            let worstIdx = -1;
            let worstSnr = Infinity;
            for (let i = 0; i < this._slots.length; i++) {
                if (!this._slots[i].isMain && this._slots[i].snr < worstSnr) {
                    worstSnr = this._slots[i].snr;
                    worstIdx = i;
                }
            }
            if (worstIdx >= 0) {
                this._destroySlot(this._slots[worstIdx]);
                this._slots.splice(worstIdx, 1);
            } else {
                break;
            }
        }
    }

    /**
     * パフォーマンス統計を取得する。
     * @returns {PerformanceStats}
     */
    get performanceStats() {
        let totalTime = 0;
        let activeCount = 0;
        let anyThrottled = false;

        for (const slot of this._slots) {
            if (slot.throttled) {
                anyThrottled = true;
                continue;
            }
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

    /**
     * 全状態をリセットする。
     */
    reset() {
        for (const slot of this._slots) {
            this._destroySlot(slot);
        }
        this._slots = [];
    }

    /**
     * 全リソースを解放する。
     */
    dispose() {
        this._disposed = true;
        this.reset();
        this._session = null;
        this._ort = null;
    }
}
