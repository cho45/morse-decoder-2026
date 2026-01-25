/**
 * DSP Utilities for CW Decoder
 */

class DSP {
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
     * Calculate Mel Filterbank
     * Matches torchaudio/librosa behavior with fractional bin interpolation and area normalization.
     */
    static createMelFilters(n_mels, n_fft, sample_rate, f_min = 500, f_max = 900) {
        const hz_to_mel = (hz) => 2595 * Math.log10(1 + hz / 700);
        const mel_to_hz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);
        
        const min_mel = hz_to_mel(f_min);
        const max_mel = hz_to_mel(f_max);
        
        const mel_points = new Float32Array(n_mels + 2);
        for (let i = 0; i < n_mels + 2; i++) {
            mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (n_mels + 1));
        }
        
        const bin_points = mel_points.map(hz => hz * n_fft / sample_rate);
        const filters = new Array(n_mels);
        const num_bins = Math.floor(n_fft / 2) + 1;
        
        for (let i = 0; i < n_mels; i++) {
            filters[i] = new Float32Array(num_bins);
            const left = bin_points[i];
            const center = bin_points[i + 1];
            const right = bin_points[i + 2];
            
            for (let j = 0; j < num_bins; j++) {
                if (j > left && j < center) {
                    filters[i][j] = (j - left) / (center - left);
                } else if (j >= center && j < right) {
                    filters[i][j] = (right - j) / (right - center);
                } else {
                    filters[i][j] = 0;
                }
            }
            
            // Area normalization
            let sum = 0;
            for (let j = 0; j < num_bins; j++) sum += filters[i][j];
            if (sum > 0) {
                for (let j = 0; j < num_bins; j++) filters[i][j] /= sum;
            }
        }
        return filters;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DSP;
}