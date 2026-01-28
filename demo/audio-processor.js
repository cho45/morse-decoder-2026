import { Resampler } from './dsp.js';

/**
 * AudioWorkletProcessor for Signal Buffering (Receiver side)
 * This processor resamples incoming audio to 16kHz and sends it to the main thread.
 */
class MorseBufferProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = options.processorOptions || {};
        this.hopLength = opts.hopLength || 160;
        
        // Resampler (sampleRate is global in AudioWorkletGlobalScope)
        this.resampler = new Resampler(sampleRate, 16000);
        
        this.audioBuffer = new Float32Array(this.hopLength);
        this.bufferPtr = 0;

        // Pre-allocate resample buffer (max expected size per process call)
        // Web Audio chunks are typically 128 samples.
        // Even at 192kHz -> 16kHz, 128 samples would result in ~11 samples.
        // We allocate 256 to be very safe and avoid allocations in process().
        this.resampleBuffer = new Float32Array(256);
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channel = input[0];
        
        // Resample the entire channel into pre-allocated buffer
        const numResampled = this.resampler.process(channel, this.resampleBuffer);

        for (let i = 0; i < numResampled; i++) {
            const sample = this.resampleBuffer[i];

            // Buffer for feature extraction (now at 16kHz)
            this.audioBuffer[this.bufferPtr++] = sample;
            if (this.bufferPtr >= this.hopLength) {
                // Send a copy of the buffer to the main thread
                this.port.postMessage({
                    type: 'audio_chunk',
                    chunk: new Float32Array(this.audioBuffer)
                });
                this.bufferPtr = 0;
            }
        }

        // Also pass through the audio to the output so it can be heard
        const output = outputs[0];
        if (output && output[0]) {
            output[0].set(channel);
        }

        return true;
    }
}

registerProcessor('morse-processor', MorseBufferProcessor);