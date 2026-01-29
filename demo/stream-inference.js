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
const DEFAULT_CHUNK_SIZE = 12; // 120ms。Must be multiple of 4 (SUBSAMPLING_RATE^2)
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

        // Clear history
        this._sigHistory = [];
        this._ctcHistory = [];
        this._boundHistory = [];
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

        this._buffer.push(new Float32Array(specFrame));
        this._totalFrames++;

        if (this._buffer.length >= this._options.chunkSize) {
            // Store the promise for later awaiting via waitForProcessing()
            this._pendingChunk = this._runChunk();
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
        if (this._buffer.length === 0) {
            return;
        }

        // Only process complete multiples of 4
        const completeChunkLen = Math.floor(this._buffer.length / 4) * 4;

        if (completeChunkLen > 0) {
            // Keep only the frames that form a complete chunk
            const remainingFrames = this._buffer.slice(completeChunkLen);
            this._buffer = this._buffer.slice(0, completeChunkLen);
            await this._runChunk();
            // Discard remaining frames (cannot be processed without padding)
            if (remainingFrames.length > 0) {
                console.warn(`StreamInference.flush(): Discarding ${remainingFrames.length} frames (not a multiple of 4)`);
            }
        } else {
            // Less than 4 frames - cannot process
            console.warn(`StreamInference.flush(): Discarding ${this._buffer.length} frames (less than 4)`);
            this._buffer = [];
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
        // Wait for current processing to complete
        while (this._isProcessing) {
            await new Promise(r => setTimeout(r, 10));
        }
        // Process any remaining buffered frames that triggered _runChunk
        if (this._pendingChunk) {
            await this._pendingChunk;
            this._pendingChunk = null;
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

    // --- Private Methods ---

    /**
     * Run inference on buffered frames.
     * @private
     */
    async _runChunk() {
        if (this._isProcessing || !this._session || !this._states) {
            return;
        }

        this._isProcessing = true;
        const chunkSize = this._buffer.length;
        const framesBefore = this._totalFrames - chunkSize;

        try {
            // Combine buffered frames into single array
            const combined = new Float32Array(N_BINS * chunkSize);
            for (let i = 0; i < chunkSize; i++) {
                combined.set(this._buffer[i], i * N_BINS);
            }
            this._buffer = [];

            // Run inference
            const result = await runChunkInference(
                this._session,
                combined,
                this._states,
                this._ort
            );

            if (this._isDisposed) {
                // If disposed during inference, update states so they can be cleaned up in finally block
                this._states = result.nextStates;
                return;
            }

            // Update states
            const oldStateValues = Object.values(this._states);
            this._states = result.nextStates;

            // Dispose old states (skip in WebGPU mode to avoid reallocation overhead)
            if (!this._options.useWebGPU) {
                const nextStateValues = Object.values(this._states);
                oldStateValues.forEach(t => {
                    if (t && t.dispose && !nextStateValues.includes(t)) {
                        try {
                            // Only dispose tensors with non-empty dims
                            if (t.dims && t.dims.length > 0 && t.dims.every(d => d > 0)) {
                                t.dispose();
                            }
                        } catch (e) {
                            // Ignore disposal errors (can happen with onnxruntime-node)
                        }
                    }
                });
            }

            // Process output frames
            const numOutFrames = result.logits.length / result.numClasses;
            const newChars = [];

            for (let t = 0; t < numOutFrames; t++) {
                // Each output frame corresponds to 2 input frames (SUBSAMPLING_RATE=2)
                const framePos = framesBefore + (t * 2);

                // Extract logits for this frame
                const ctcLogits = result.logits.slice(t * result.numClasses, (t + 1) * result.numClasses);
                const sigLogits = result.signalLogits.slice(t * 4, (t + 1) * 4);
                const boundLogit = result.boundaryLogits[t];

                // Calculate probabilities
                const ctcProbs = softmax(Array.from(ctcLogits));
                const sigProbs = softmax(Array.from(sigLogits));
                const boundProb = sigmoid(boundLogit);

                // Update history
                this._sigHistory.push({ probs: sigProbs, pos: framePos });
                this._ctcHistory.push(ctcProbs);
                this._boundHistory.push(boundProb);

                // Trim history to max length
                while (this._sigHistory.length > this._options.historyLength) {
                    this._sigHistory.shift();
                }
                while (this._ctcHistory.length > this._options.historyLength) {
                    this._ctcHistory.shift();
                }
                while (this._boundHistory.length > this._options.historyLength) {
                    this._boundHistory.shift();
                }

                // Decode frame
                const decodeResult = this._decoder.decodeFrame(ctcLogits, sigLogits, boundProb);

                // Track new characters
                if (decodeResult.newChar) {
                    if (decodeResult.spaceInserted) {
                        this._events.push({ char: ' ', pos: framePos - 1 });
                        newChars.push(' ');
                    }
                    this._events.push({ char: decodeResult.newChar, pos: framePos });
                    newChars.push(decodeResult.newChar);
                }
            }

            // Trim events to reasonable length
            // historyLength is in output frames, but pos/totalFrames are in input frames (2x)
            const historyInInputFrames = this._options.historyLength * 2;
            while (this._events.length > 0 && this._events[0].pos < this._totalFrames - historyInInputFrames) {
                this._events.shift();
            }

            // Fire events
            if (newChars.length > 0) {
                this.dispatchEvent(new CustomEvent('result', {
                    detail: {
                        text: this._decoder.getText(),
                        newChars: newChars,
                        framePos: this._totalFrames
                    }
                }));
            }

            this.dispatchEvent(new CustomEvent('frame', {
                detail: {
                    framePos: this._totalFrames
                }
            }));

        } catch (e) {
            console.error('StreamInference error:', e);
            throw e;
        } finally {
            this._isProcessing = false;
            if (this._isDisposed) {
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
