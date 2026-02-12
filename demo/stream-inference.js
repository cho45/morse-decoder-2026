/**
 * StreamInference - Stateful streaming inference manager for CW decoding.
 *
 * Encapsulates:
 * - Model state management (ONNX tensor states)
 * - Frame buffering until chunk size is reached
 * - CTC decoding via ChunkedDecoder
 * - Visualization history management
 * - Event-based result notification
 */

import {
    N_BINS, NUM_LAYERS, D_MODEL, N_HEAD, D_K, KERNEL_SIZE,
    CHARS, ID_TO_CHAR,
    softmax, sigmoid, initStates, runChunkInference, ChunkedDecoder
} from './inference.js';

// Default configuration
// 【重要】DEFAULT_CHUNK_SIZE は必ず 4 の倍数（SUBSAMPLING_RATE=2 の二乗）である必要があります。
// 4 の倍数でないチャンクを入力すると、runChunkInference がエラーを投げます。
// これは ONNX Runtime の制限によるものです：state updates が output より前に実行されるため、
// パディングを行うとモデルの内部キャッシュ（ConvSubsampling, ConformerConvModule, Attention）が
// 破損してしまいます。
const DEFAULT_CHUNK_SIZE = 20; // 200ms。Must be multiple of 4 (SUBSAMPLING_RATE^2)
const DEFAULT_HISTORY_LENGTH = 800;

/**
 * StreamInference class for managing real-time CW decoder inference.
 * Extends EventTarget to emit 'result' and 'frame' events.
 *
 * Events:
 * - 'result': Fired when new characters are decoded
 *   - detail.text: Full decoded text
 *   - detail.newChars: Array of newly decoded characters
 *   - detail.framePos: Current frame position
 *   - detail.spaceInserted: Whether a space was inserted before the character
 *
 * - 'frame': Fired after each inference chunk is processed
 *   - detail.framePos: Current frame position
 *
 * @example
 * const inference = new StreamInference(session, ort, { chunkSize: 12 });
 * inference.addEventListener('result', (e) => {
 *     console.log('Decoded:', e.detail.text);
 * });
 * // Feed frames from audio processor
 * inference.pushFrame(specFrame);
 */
export class StreamInference extends EventTarget {
    /**
     * @param {ort.InferenceSession} session - ONNX Runtime inference session
     * @param {typeof ort} ort - ONNX Runtime module
     * @param {Object} options - Configuration options
     * @param {number} [options.chunkSize=12] - Number of frames per inference chunk (must be multiple of 4)
     * @param {boolean} [options.useWebGPU=false] - Whether to use WebGPU (affects tensor disposal strategy)
     * @param {number} [options.historyLength=800] - Maximum history length for visualization data
     */
    constructor(session, ort, options = {}) {
        super();

        if (!session) {
            throw new Error('StreamInference requires an ONNX session');
        }
        if (!ort) {
            throw new Error('StreamInference requires ort module');
        }

        this._session = session;
        this._ort = ort;
        this._options = {
            chunkSize: options.chunkSize || DEFAULT_CHUNK_SIZE,
            useWebGPU: options.useWebGPU || false,
            historyLength: options.historyLength || DEFAULT_HISTORY_LENGTH
        };

        // Validate chunk size
        if (this._options.chunkSize % 4 !== 0) {
            throw new Error(`chunkSize must be a multiple of 4, got ${this._options.chunkSize}`);
        }

        // Initialize internal state
        this._states = null;
        this._decoder = null;
        this._buffer = [];
        this._totalFrames = 0;
        this._events = [];
        this._isProcessing = false;
        this._pendingChunk = null;
        this._isDisposed = false;

        // Visualization history
        this._sigHistory = [];
        this._ctcHistory = [];
        this._boundHistory = [];

        // Performance metrics
        this._lastTensorTime = 0;
        this._lastSessionTime = 0;

        // Initialize
        this.reset();
    }

    /**
     * Reset all internal state to initial values.
     * Call this when starting a new decoding session.
     */
    reset() {
        // Dispose existing states if any
        this._disposeStates();

        // Initialize fresh states
        this._states = initStates(this._ort);
        this._decoder = new ChunkedDecoder(CHARS.length + 1);
        this._buffer = [];
        this._totalFrames = 0;
        this._events = [];
        this._isProcessing = false;
        this._pendingChunk = null;
        this._processedFrames = 0;

        // Clear history
        this._sigHistory = [];
        this._ctcHistory = [];
        this._boundHistory = [];

        // Reset metrics
        this._lastTensorTime = 0;
        this._lastSessionTime = 0;
    }

    /**
     * Push a single spectrogram frame for inference.
     * When buffer reaches chunkSize, inference is automatically triggered.
     *
     * @param {Float32Array} specFrame - Spectrogram frame [N_BINS]
     */
    pushFrame(specFrame) {
        if (specFrame.length !== N_BINS) {
            throw new Error(`Expected frame of ${N_BINS} bins, got ${specFrame.length}`);
        }
        this._buffer.push(specFrame);
        this._totalFrames++;

        if (!this._isProcessing) {
            this._runChunk();
        }
    }

    /**
     * Process any buffered frames that form a complete chunk (multiple of 4).
     * Frames that don't complete a chunk are discarded with a warning.
     *
     * NOTE: Zero-padding is NOT used because it corrupts the model's internal cache
     * (ConvSubsampling, ConformerConvModule, Attention). This is an ONNX Runtime
     * limitation where state updates happen before output is returned.
     *
     * @returns {Promise<void>}
     */
    async flush() {
        // Wait for any ongoing processing to complete
        await this.waitForProcessing();

        if (this._buffer.length === 0) {
            return;
        }

        // Only process complete multiples of 4
        const completeChunkLen = Math.floor(this._buffer.length / 4) * 4;

        if (completeChunkLen > 0) {
            // Keep only the frames that form a complete chunk
            const framesToProcess = this._buffer.splice(0, completeChunkLen);

            // Temporarily store current buffer, process framesToProcess, then restore
            const originalBuffer = this._buffer;
            this._buffer = framesToProcess;

            await this._runChunk(true);

            this._buffer = originalBuffer; // Restore remaining frames
            // Frames remaining in buffer will be processed next time (or require more data)
        } else {
            // Less than 4 frames - cannot process
            // leave in buffer (do not discard)
        }
    }

    /**
     * Get current decoded text.
     * @returns {string}
     */
    getText() {
        return this._decoder ? this._decoder.getText() : '';
    }

    /**
     * Get decoded character events with positions.
     * @returns {Array<{char: string, pos: number}>}
     */
    getEvents() {
        return this._events.slice();
    }

    /**
     * Get signal probability history for visualization.
     * @returns {Array<{probs: number[], pos: number}>}
     */
    getSignalHistory() {
        return this._sigHistory.slice();
    }

    /**
     * Get CTC probability history for visualization.
     * @returns {Array<number[]>}
     */
    getCTCHistory() {
        return this._ctcHistory.slice();
    }

    /**
     * Get boundary probability history for visualization.
     * @returns {Array<number>}
     */
    getBoundaryHistory() {
        return this._boundHistory.slice();
    }

    /**
     * Release all resources.
     * Call this when done with the instance.
     */
    dispose() {
        this._isDisposed = true;
        if (this._isProcessing) {
            // Defer state disposal until inference completes
            // We still clear other resources to stop further usage
            this._decoder = null;
            this._buffer = [];
            this._events = [];
            this._sigHistory = [];
            this._ctcHistory = [];
            this._boundHistory = [];
            this._session = null;
            this._ort = null;
            return;
        }
        this._disposeStates();
        this._decoder = null;
        this._buffer = [];
        this._events = [];
        this._sigHistory = [];
        this._ctcHistory = [];
        this._boundHistory = [];
        this._session = null;
        this._ort = null;
    }

    /**
     * Wait for all pending inference to complete.
     * @returns {Promise<void>}
     */
    async waitForProcessing() {
        // Wait for current processing chain to complete
        while (this._isProcessing) {
            await this._processingPromise;
        }
    }

    /**
     * Whether inference is currently running.
     * @returns {boolean}
     */
    get isProcessing() {
        return this._isProcessing;
    }

    /**
     * Current frame count.
     * @returns {number}
     */
    get frameCount() {
        return this._totalFrames;
    }

    /**
     * Number of frames currently buffered.
     * @returns {number}
     */
    get bufferSize() {
        return this._buffer.length;
    }

    /**
     * Last inference session time in ms.
     * @returns {number}
     */
    get inferenceTime() {
        return this._lastSessionTime;
    }

    /**
     * Last tensor preparation (normalization) time in ms.
     * @returns {number}
     */
    get tensorTime() {
        return this._lastTensorTime;
    }

    // --- Private Methods ---

    /**
     * Run inference on buffered frames.
     * @param {boolean} [flush=false] - If true, process even if buffer < chunkSize (multiple of 4)
     * @private
     */
    async _runChunk(flush = false) {
        if (this._isProcessing || !this._session || !this._states) {
            return;
        }

        // Guard: check if we have enough data to process
        // We need at least chunkSize, OR if flushing, at least 4 frames (ONNX requirement)
        const minFrames = flush ? 4 : this._options.chunkSize;
        if (this._buffer.length < minFrames) {
            return;
        }

        this._isProcessing = true;
        let resolveProcessing;
        this._processingPromise = new Promise(r => resolveProcessing = r);

        try {
            // Determine actual chunk size to process
            let processSize = this._options.chunkSize;
            if (flush && this._buffer.length < processSize) {
                // If flushing and less than chunkSize, take largest multiple of 4
                processSize = Math.floor(this._buffer.length / 4) * 4;
            }

            // Extract exact chunk
            if (processSize === 0) return;
            const chunk = this._buffer.splice(0, processSize);

            // Should not happen due to guard, but check
            if (chunk.length === 0) return;

            // Combine into input tensor
            const inputTensor = new Float32Array(N_BINS * chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                inputTensor.set(chunk[i], i * N_BINS);
            }

            // Run inference
            const result = await runChunkInference(
                this._session,
                inputTensor,
                this._states,
                this._ort
            );

            if (this._isDisposed) {
                this._states = result.nextStates;
                return;
            }

            // Update states and metrics
            const oldStateValues = Object.values(this._states);
            this._states = result.nextStates;
            this._lastTensorTime = result.tensorTime;
            this._lastSessionTime = result.sessionTime;

            // Dispose old states
            if (!this._options.useWebGPU) {
                const nextStateValues = Object.values(this._states);
                oldStateValues.forEach(t => {
                    if (t && t.dispose && !nextStateValues.includes(t)) {
                        try {
                            if (t.dims && t.dims.length > 0 && t.dims.every(d => d > 0)) {
                                t.dispose();
                            }
                        } catch (e) { }
                    }
                });
            }

            // Process outputs
            const numOutFrames = result.logits.length / result.numClasses;
            const newChars = [];

            // Frame position calculation:
            // Since we process sequentially, `_processedFrames` tracks the start of this chunk.
            const chunkStartPos = this._processedFrames;

            for (let t = 0; t < numOutFrames; t++) {
                // Output corresponds to 2 input frames (subsampling)
                const framePos = chunkStartPos + (t * 2);

                const ctcLogits = result.logits.slice(t * result.numClasses, (t + 1) * result.numClasses);
                const sigLogits = result.signalLogits.slice(t * 4, (t + 1) * 4);
                const boundLogit = result.boundaryLogits[t];

                const ctcProbs = softmax(Array.from(ctcLogits));
                const sigProbs = softmax(Array.from(sigLogits));
                const boundProb = sigmoid(boundLogit);

                this._sigHistory.push({ probs: sigProbs, pos: framePos });
                this._ctcHistory.push(ctcProbs);
                this._boundHistory.push(boundProb);

                while (this._sigHistory.length > this._options.historyLength) this._sigHistory.shift();
                while (this._ctcHistory.length > this._options.historyLength) this._ctcHistory.shift();
                while (this._boundHistory.length > this._options.historyLength) this._boundHistory.shift();

                const decodeResult = this._decoder.decodeFrame(ctcLogits, sigLogits, boundProb);
                if (decodeResult.newChar) {
                    this._events.push({ char: decodeResult.newChar, pos: framePos });
                    newChars.push(decodeResult.newChar);
                }
            }

            this._processedFrames += chunk.length;

            // Trim events
            const historyInInputFrames = this._options.historyLength * 2;
            while (this._events.length > 0 && this._events[0].pos < this._processedFrames - historyInInputFrames) {
                this._events.shift();
            }

            // Emit result
            if (newChars.length > 0) {
                this.dispatchEvent(new CustomEvent('result', {
                    detail: {
                        text: this._decoder.getText(),
                        newChars: newChars,
                        framePos: this._processedFrames
                    }
                }));
            }

            this.dispatchEvent(new CustomEvent('frame', {
                detail: {
                    framePos: this._processedFrames
                }
            }));

        } catch (e) {
            console.error('StreamInference error:', e);
            throw e;
        } finally {
            this._isProcessing = false;
            if (resolveProcessing) resolveProcessing();

            // Recursive chain: check if more processing needed
            if (!this._isDisposed) {
                const nextMinFrames = flush ? 4 : this._options.chunkSize;
                if (this._buffer.length >= nextMinFrames) {
                    this._runChunk(flush);
                }
            } else {
                this._disposeStates();
            }
        }
    }

    /**
     * Dispose ONNX tensor states.
     * @private
     */
    _disposeStates() {
        if (this._states && !this._options.useWebGPU) {
            Object.values(this._states).forEach(t => {
                if (t && t.dispose && t.dims && t.dims.length > 0 && t.dims.every(d => d > 0)) {
                    try {
                        t.dispose();
                    } catch (e) {
                        // Ignore disposal errors
                    }
                }
            });
        }
        this._states = null;
    }
}

// Re-export useful constants and utilities for convenience
export { N_BINS, CHARS, ID_TO_CHAR, softmax, sigmoid };
