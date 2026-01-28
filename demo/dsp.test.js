import { describe, it, expect } from 'vitest';
import { DSP, Resampler } from './dsp.js';

describe('DSP class', () => {
    describe('fft', () => {
        it('should compute FFT of a simple impulse', () => {
            const size = 8;
            const real = new Float32Array(size);
            const imag = new Float32Array(size);
            real[0] = 1.0;

            DSP.fft(real, imag);

            // FFT of [1, 0, 0, 0, 0, 0, 0, 0] is [1, 1, 1, 1, 1, 1, 1, 1]
            for (let i = 0; i < size; i++) {
                expect(real[i]).toBeCloseTo(1.0);
                expect(imag[i]).toBeCloseTo(0.0);
            }
        });

        it('should compute FFT of a sine wave', () => {
            const size = 16;
            const real = new Float32Array(size);
            const imag = new Float32Array(size);
            const freq = 2; // 2 cycles in 16 samples
            for (let i = 0; i < size; i++) {
                real[i] = Math.cos(2 * Math.PI * freq * i / size);
            }

            DSP.fft(real, imag);

            // Peak should be at index 2 (and size-2 due to symmetry)
            // Magnitude at peak should be size/2 = 8
            expect(Math.sqrt(real[2]**2 + imag[2]**2)).toBeCloseTo(size / 2);
            expect(Math.sqrt(real[size - freq]**2 + imag[size - freq]**2)).toBeCloseTo(size / 2);
            
            // Other bins should be near zero
            for (let i = 0; i < size; i++) {
                if (i !== freq && i !== size - freq) {
                    expect(Math.sqrt(real[i]**2 + imag[i]**2)).toBeLessThan(1e-5);
                }
            }
        });

        it('should throw error if size is not a power of 2', () => {
            const real = new Float32Array(10);
            const imag = new Float32Array(10);
            expect(() => DSP.fft(real, imag)).toThrow("FFT size must be a power of 2");
        });

        it('should detect correct frequency like in test_dsp.js', () => {
            const n = 512;
            const real = new Float32Array(n);
            const imag = new Float32Array(n);
            const freq = 1000;
            const sr = 16000;
            for (let i = 0; i < n; i++) {
                real[i] = Math.sin(2 * Math.PI * freq * i / sr);
            }
            DSP.fft(real, imag);
            let maxMag = 0;
            let maxIdx = 0;
            for (let i = 0; i < n / 2; i++) {
                const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
                if (mag > maxMag) {
                    maxMag = mag;
                    maxIdx = i;
                }
            }
            const detectedFreq = maxIdx * sr / n;
            expect(Math.abs(detectedFreq - freq)).toBeLessThan(sr / n);
        });
    });

    describe('lfilter', () => {
        it('should apply a simple moving average filter', () => {
            const b = [0.5, 0.5];
            const a = [1.0];
            const x = new Float32Array([1, 0, 0, 0]);
            const y = DSP.lfilter(b, a, x);

            // y[0] = 0.5*x[0] + 0.5*x[-1] = 0.5
            // y[1] = 0.5*x[1] + 0.5*x[0] = 0.5
            // y[2] = 0.5*x[2] + 0.5*x[1] = 0
            expect(y[0]).toBeCloseTo(0.5);
            expect(y[1]).toBeCloseTo(0.5);
            expect(y[2]).toBeCloseTo(0.0);
        });

        it('should handle normalization by a[0]', () => {
            const b = [1.0];
            const a = [2.0]; // Gain should be 0.5
            const x = new Float32Array([1, 1, 1]);
            const y = DSP.lfilter(b, a, x);

            expect(y[0]).toBeCloseTo(0.5);
            expect(y[1]).toBeCloseTo(0.5);
            expect(y[2]).toBeCloseTo(0.5);
        });

        it('should handle empty input', () => {
            const b = [1.0];
            const a = [1.0];
            const x = new Float32Array(0);
            const y = DSP.lfilter(b, a, x);
            expect(y.length).toBe(0);
        });

        it('should handle single element input', () => {
            const b = [1.0];
            const a = [1.0];
            const x = new Float32Array([5.0]);
            const y = DSP.lfilter(b, a, x);
            expect(y[0]).toBe(5.0);
        });
    });

    describe('generateGaussianNoise', () => {
        it('should generate noise with correct length', () => {
            const length = 1000;
            const noise = DSP.generateGaussianNoise(length, 1.0);
            expect(noise.length).toBe(length);
        });

        it('should have approximately zero mean and correct sigma', () => {
            const length = 10000;
            const sigma = 2.5;
            const noise = DSP.generateGaussianNoise(length, sigma);

            let sum = 0;
            for (let v of noise) sum += v;
            const mean = sum / length;

            let sumSq = 0;
            for (let v of noise) sumSq += (v - mean) ** 2;
            const calculatedSigma = Math.sqrt(sumSq / length);

            expect(mean).toBeCloseTo(0.0, 0); // Be more lenient: 0.1 is fine for 10k samples
            expect(Math.abs(mean)).toBeLessThan(0.15);
            expect(calculatedSigma).toBeCloseTo(sigma, 0);
            expect(Math.abs(calculatedSigma - sigma)).toBeLessThan(0.15);
        });
    });
});

describe('Resampler class', () => {
    it('should maintain DC offset (1.0 in -> 1.0 out) for integer ratio', () => {
        const resampler = new Resampler(48000, 16000); // Ratio 3
        const input = new Float32Array(30).fill(1.0);
        const output = new Float32Array(10);
        const n = resampler.process(input, output);
        
        expect(n).toBe(10);
        for (let v of output) expect(v).toBeCloseTo(1.0);
    });

    it('should maintain DC offset for non-integer ratio (44.1k -> 16k)', () => {
        const resampler = new Resampler(44100, 16000); // Ratio 2.75625
        const input = new Float32Array(4410).fill(1.0); // 0.1s
        const output = new Float32Array(1600);
        const n = resampler.process(input, output);
        
        expect(n).toBe(1600);
        for (let v of output) expect(v).toBeCloseTo(1.0);
    });

    it('should handle streaming correctly', () => {
        const resampler = new Resampler(48000, 16000); // Ratio 3
        const outBuf = new Float32Array(1);
        
        const n1 = resampler.process(new Float32Array([1, 1]), outBuf);
        expect(n1).toBe(0);
        
        const n2 = resampler.process(new Float32Array([1, 1]), outBuf);
        expect(n2).toBe(1);
        expect(outBuf[0]).toBeCloseTo(1.0);
        
        const n3 = resampler.process(new Float32Array([1, 1]), outBuf);
        expect(n3).toBe(1);
        expect(outBuf[0]).toBeCloseTo(1.0);
    });

    it('should handle sine wave frequency correctly', () => {
        const srIn = 48000;
        const srOut = 16000;
        const freq = 1000;
        const resampler = new Resampler(srIn, srOut);
        
        const input = new Float32Array(srIn / 10); // 0.1s
        for (let i = 0; i < input.length; i++) {
            input[i] = Math.sin(2 * Math.PI * freq * i / srIn);
        }
        
        const output = new Float32Array(srOut / 10);
        const n = resampler.process(input, output);
        expect(n).toBe(srOut / 10);
        
        let peaks = 0;
        for (let i = 1; i < output.length - 1; i++) {
            if (output[i] > output[i-1] && output[i] > output[i+1] && output[i] > 0.5) {
                peaks++;
            }
        }
        expect(peaks).toBeGreaterThan(95);
        expect(peaks).toBeLessThan(105);
    });
});