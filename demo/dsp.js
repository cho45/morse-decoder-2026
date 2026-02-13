/**
 * DSP Utilities for CW Decoder
 */

export class DSP {
    /**
     * Fast Fourier Transform (Radix-2 Cooley-Tukey)
     * Input size must be a power of 2.
     */
    static fft(real, imag) {
        const n = real.length;
        if ((n & (n - 1)) !== 0) throw new Error("FFT size must be a power of 2");

        for (let i = 0, j = 0; i < n; i++) {
            if (i < j) {
                [real[i], real[j]] = [real[j], real[i]];
                [imag[i], imag[j]] = [imag[j], imag[i]];
            }
            let m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        for (let len = 2; len <= n; len <<= 1) {
            const ang = 2 * Math.PI / len;
            const wlen_real = Math.cos(ang);
            const wlen_imag = -Math.sin(ang);
            for (let i = 0; i < n; i += len) {
                let w_real = 1;
                let w_imag = 0;
                for (let j = 0; j < len / 2; j++) {
                    const u_real = real[i + j];
                    const u_imag = imag[i + j];
                    const v_real = real[i + j + len / 2] * w_real - imag[i + j + len / 2] * w_imag;
                    const v_imag = real[i + j + len / 2] * w_imag + imag[i + j + len / 2] * w_real;
                    real[i + j] = u_real + v_real;
                    imag[i + j] = u_imag + v_imag;
                    real[i + j + len / 2] = u_real - v_real;
                    imag[i + j + len / 2] = u_imag - v_imag;
                    const next_w_real = w_real * wlen_real - w_imag * wlen_imag;
                    w_imag = w_real * wlen_imag + w_imag * wlen_real;
                    w_real = next_w_real;
                }
            }
        }
    }

    /**
     * IIR Filter (Direct Form II Transposed)
     * Equivalent to scipy.signal.lfilter(b, a, x)
     */
    static lfilter(b, a, x) {
        const n = x.length;
        const nb = b.length;
        const na = a.length;
        // Use Float64Array for internal state to maintain precision for high-order filters
        const y = new Float64Array(n);
        const z = new Float64Array(Math.max(nb, na));

        // Normalize by a[0]
        const a0 = a[0];
        const nb_norm = b.map(v => v / a0);
        const na_norm = a.map(v => v / a0);

        for (let i = 0; i < n; i++) {
            y[i] = nb_norm[0] * x[i] + z[0];
            for (let j = 1; j < Math.max(nb, na); j++) {
                const bj = j < nb ? nb_norm[j] : 0;
                const aj = j < na ? na_norm[j] : 0;
                z[j - 1] = bj * x[i] - aj * y[i] + (z[j] || 0);
            }
        }
        // Convert back to Float32Array if input was Float32Array, or return Float64Array
        if (x instanceof Float32Array) {
            return new Float32Array(y);
        }
        return y;
    }

    /**
     * Generate Gaussian Noise (Box-Muller transform)
     */
    static generateGaussianNoise(length, sigma = 1.0) {
        const noise = new Float32Array(length);
        for (let i = 0; i < length; i += 2) {
            // Avoid u1 = 0 to prevent log(0) = -Infinity
            const u1 = 1.0 - Math.random();
            const u2 = Math.random();
            const mag = sigma * Math.sqrt(-2.0 * Math.log(u1));
            noise[i] = mag * Math.cos(2.0 * Math.PI * u2);
            if (i + 1 < length) {
                noise[i + 1] = mag * Math.sin(2.0 * Math.PI * u2);
            }
        }
        return noise;
    }
}

/**
 * Resampler with Anti-Aliasing (Box Filter / Integrator)
 * Efficiently downsamples audio while suppressing aliasing and jitter.
 */
export class Resampler {
    /**
     * @param {number} sourceRate - Input sample rate (e.g., 44100)
     * @param {number} targetRate - Output sample rate (e.g., 16000)
     */
    constructor(sourceRate, targetRate) {
        this.sourceRate = sourceRate;
        this.targetRate = targetRate;
        this.ratio = sourceRate / targetRate;
        this.accum = 0;
        this.timeLeft = this.ratio;

        // --- NEW: Polyphase FIR Coefficient Generation ---
        // For 48k -> 32k, ratio is 1.5 (3:2). 
        // We'll use a fixed number of phases (oversampling) to approximate the ratio,
        // or strictly follow the rational ratio if possible.
        // For simplicity and high quality, let's use a large oversampling (e.g. 64 phases)
        // for generic ratios, providing quasi-continuous delay-line sinc interpolation.
        this.numPhases = 64;
        this.tapsPerPhase = 12;
        this.coeffs = new Array(this.numPhases);

        const cutoff = Math.min(1.0, 1.0 / this.ratio) * 0.9; // Anti-aliasing cutoff
        const halfSize = this.tapsPerPhase / 2;

        for (let p = 0; p < this.numPhases; p++) {
            const phase = p / this.numPhases;
            const phaseCoeffs = new Float32Array(this.tapsPerPhase);
            let sum = 0;

            for (let i = 0; i < this.tapsPerPhase; i++) {
                // Sinc center is at p/numPhases
                const x = (i - halfSize + 1 - phase) * cutoff;

                // Sinc * Blackman window
                let weight = 1.0;
                if (Math.abs(x) > 1e-10) {
                    const piX = Math.PI * x;
                    weight = Math.sin(piX) / piX;
                }

                // Window (Blackman)
                const t = (i - phase) / (this.tapsPerPhase - 1);
                const blackman = 0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t);
                weight *= blackman;

                phaseCoeffs[i] = weight;
                sum += weight;
            }

            // Normalize for DC gain = 1.0
            for (let i = 0; i < this.tapsPerPhase; i++) {
                phaseCoeffs[i] /= sum;
            }
            this.coeffs[p] = phaseCoeffs;
        }

        // Initialize pointer to wait for enough samples, matching Box filter's output timing.
        this.fractionalIndex = this.ratio - 1;
    }

    /**
     * Process a chunk of samples and write resampled samples to output array.
     * @param {Float32Array} input - Input samples
     * @param {Float32Array} output - Output buffer
     * @returns {number} Number of samples written to output
     */
    process(input, output) {
        const halfSize = Math.floor(this.tapsPerPhase / 2);

        // 1. Initialize history on first call
        if (!this.history) {
            this.history = new Float32Array(halfSize);
            // Pre-fill with first sample to avoid slow ramp-up in DC tests
            if (input.length > 0) {
                this.history.fill(input[0]);
            }
        }

        let outPtr = 0;
        const inputLen = input.length;

        // Use a temporary buffer combining history and current input
        // For FIR windowed sinc, we need some 'history' samples.
        const buffer = new Float32Array(halfSize + inputLen + halfSize);
        buffer.set(this.history);
        buffer.set(input, halfSize);
        // Fill future padding with the last sample to prevent artifacts at the very end
        if (inputLen > 0) {
            buffer.fill(input[inputLen - 1], halfSize + inputLen);
        }

        let idx = this.fractionalIndex || 0; // index relative to the start of 'input'

        while (idx < inputLen && outPtr < output.length) {
            const intPart = Math.floor(idx);
            const frac = idx - intPart;
            // Phase selection
            const phaseIdx = Math.min(this.numPhases - 1, Math.floor(frac * this.numPhases));
            const coeffs = this.coeffs[phaseIdx];

            let sum = 0;
            // Apply FIR filter: centered at idx.
            // idx in 'input' maps to 'idx + halfSize' in 'buffer'.
            // Kernel covers buffer[intPart + i] where i=0...taps-1.
            // If intPart=0, it uses buffer[0...11]. idx=0 is at buffer[6].
            // So it uses 6 samples before and 6 samples at/after idx.
            for (let i = 0; i < this.tapsPerPhase; i++) {
                sum += buffer[intPart + i] * coeffs[i];
            }
            output[outPtr++] = sum;
            idx += this.ratio;
        }

        // Save state for next call
        this.fractionalIndex = idx - inputLen;
        if (inputLen >= halfSize) {
            this.history.set(input.subarray(inputLen - halfSize));
        } else if (inputLen > 0) {
            // Shift existing history and append new input
            this.history.copyWithin(0, inputLen);
            this.history.set(input, halfSize - inputLen);
        }

        return outPtr;
    }
}