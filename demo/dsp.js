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
            const u1 = Math.random();
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