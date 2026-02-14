import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js";
import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";
import { initStates, runChunkInference } from '../inference.js';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

class ModelWorker {
    constructor() {
        this.session = null;
        this.states = null;
    }

    async init(modelBuffer) {
        try {
            // ONNX Runtime can take ArrayBuffer directly
            this.session = await ort.InferenceSession.create(modelBuffer);
            this.states = initStates(ort);
            return true;
        } catch (e) {
            console.error("Worker init error:", e);
            throw e;
        }
    }

    async infer(chunkFrames) {
        if (!this.session) throw new Error('Session not initialized');

        const start = performance.now();
        const result = await runChunkInference(this.session, chunkFrames, this.states, ort);
        const end = performance.now();

        this.states = result.nextStates;

        const transferables = [
            result.logits.buffer,
            result.signalLogits.buffer,
            result.boundaryLogits.buffer
        ];

        return Comlink.transfer({
            logits: result.logits,
            signalLogits: result.signalLogits,
            boundaryLogits: result.boundaryLogits,
            numClasses: result.numClasses,
            time: end - start
        }, transferables);
    }
}

Comlink.expose(new ModelWorker());
