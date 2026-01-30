import { describe, it, expect } from 'vitest';
import { HFChannelSimulator } from './data_gen.js';

describe('SNR Definition Independence (JS)', () => {
    function calculateN0(waveform, sampleRate) {
        // 全帯域電力 (分散)
        let sum = 0;
        let sumSq = 0;
        for (let i = 0; i < waveform.length; i++) {
            sum += waveform[i];
            sumSq += waveform[i] * waveform[i];
        }
        const mean = sum / waveform.length;
        const variance = (sumSq / waveform.length) - (mean * mean);
        
        // 1Hzあたりの密度 N0 = P / (Fs/2)
        return variance / (sampleRate / 2);
    }

    it('should have sampling rate independent noise density (snr_2500)', () => {
        const snr_2500 = -10.0;
        const fsList = [8000, 16000, 48000];
        const n0Values = [];

        for (const fs of fsList) {
            const sim = new HFChannelSimulator(fs);
            const waveform = new Float32Array(fs).fill(0); // 1.0s duration
            
            // 修正後のインターフェース
            const noisy = sim.applyNoise(waveform, { snr_2500 });
            
            const n0 = calculateN0(noisy, fs);
            n0Values.push(n0);
            console.log(`Fs: ${fs}Hz, N0: ${n0.toExponential(10)}`);
        }

        // 現状の実装では Fs が大きくなると N0 が下がるため、ここで失敗するはず
        for (let i = 0; i < n0Values.length - 1; i++) {
            // モンテカルロ的な揺らぎを考慮し、相対誤差 10% 程度を許容
            const relDiff = Math.abs(n0Values[i] - n0Values[i+1]) / n0Values[i];
            expect(relDiff).toBeLessThan(0.1);
        }
    });
});