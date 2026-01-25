/**
 * AudioWorkletProcessor for Signal Buffering (Receiver side)
 * This processor only buffers incoming audio and sends it to the main thread for inference.
 */
class MorseBufferProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.hopLength = 160; // 10ms @ 16kHz
        this.audioBuffer = new Float32Array(this.hopLength);
        this.bufferPtr = 0;
    }

    process(inputs, outputs, parameters) {
        // We take input from the first node connected to us (the mix of Morse and Noise)
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channel = input[0];
        const numSamples = channel.length;

        for (let i = 0; i < numSamples; i++) {
            const sample = channel[i];

            // Buffer for feature extraction
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