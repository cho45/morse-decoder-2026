import { describe, it, expect } from 'vitest';
import { MorseGenerator, HFChannelSimulator, MORSE_DICT } from './data_gen.js';

describe('MorseGenerator', () => {
    const sampleRate = 16000;
    const gen = new MorseGenerator(sampleRate);

    it('should tokenize text with prosigns correctly', () => {
        const tokens = gen.textToMorseTokens('CQ <BT> TEST');
        expect(tokens).toContain('<BT>');
        expect(tokens).toEqual(['C', 'Q', ' ', '<BT>', ' ', 'T', 'E', 'S', 'T']);
    });

    it('should generate timing for "E"', () => {
        const timing = gen.generateTiming('E', 20);
        // 'E' is '.', so timing should be [{classId: 1, duration: dotLen}]
        expect(timing.length).toBe(1);
        expect(timing[0].classId).toBe(1);
        expect(timing[0].duration).toBeCloseTo(1.2 / 20, 5);
    });

    it('should generate timing for "CQ"', () => {
        const timing = gen.generateTiming('CQ', 20);
        // C: -.-. (4 elements + 3 intra-char spaces)
        // Inter-char space (1)
        // Q: --.- (4 elements + 3 intra-char spaces)
        // Total: 7 + 1 + 7 = 15 elements
        expect(timing.length).toBe(15);
    });

    it('should generate waveform for "E" with correct envelope and duration', () => {
        const wpm = 20;
        const freq = 700;
        const riseTime = 0.005;
        const timing = gen.generateTiming('E', wpm);
        const waveform = gen.generateWaveform(timing, freq, riseTime);
        
        const dotLenSamples = Math.floor((1.2 / wpm) * sampleRate);
        const preSilenceSamples = Math.floor(0.2 * sampleRate);
        const postSilenceSamples = Math.floor(0.55 * sampleRate);
        const expectedLength = dotLenSamples + preSilenceSamples + postSilenceSamples;
        
        // Allow +- 1 sample diff
        expect(Math.abs(waveform.length - expectedLength)).toBeLessThanOrEqual(1);
        
        // Check pre-silence
        for (let i = 0; i < preSilenceSamples - 10; i++) {
            expect(waveform[i]).toBe(0);
        }
        
        // Check signal start (rise envelope)
        const signalStart = preSilenceSamples;
        expect(Math.abs(waveform[signalStart])).toBeLessThan(0.1);
        
        // Check middle of the dot (should be near peak of sine)
        // We look for the maximum value in the middle range to ensure it reached ~1.0
        let maxVal = 0;
        for (let i = signalStart + 10; i < signalStart + dotLenSamples - 10; i++) {
            maxVal = Math.max(maxVal, Math.abs(waveform[i]));
        }
        expect(maxVal).toBeGreaterThan(0.9);
        
        // Check signal end (fall envelope)
        const signalEnd = signalStart + dotLenSamples;
        expect(Math.abs(waveform[signalEnd - 1])).toBeLessThan(0.1);
        
        // Check post-silence
        for (let i = signalEnd + 10; i < waveform.length; i++) {
            expect(waveform[i]).toBe(0);
        }
    });

    it('should handle empty string', () => {
        const timing = gen.generateTiming('', 20);
        expect(timing).toEqual([]);
        const waveform = gen.generateWaveform(timing);
        expect(waveform.length).toBeGreaterThan(0); // pre/post silence
        waveform.forEach(v => expect(v).toBe(0));
    });

    it('should handle unknown characters', () => {
        const timing = gen.generateTiming('!', 20);
        expect(timing).toEqual([]);
    });

    it('should handle unclosed prosigns', () => {
        const tokens = gen.textToMorseTokens('<BT');
        expect(tokens).toEqual(['<', 'B', 'T']);
    });

    it('should handle extreme WPM', () => {
        const timingLow = gen.generateTiming('E', 1);
        const timingHigh = gen.generateTiming('E', 100);
        expect(timingLow[0].duration).toBeGreaterThan(timingHigh[0].duration);
    });

    it('should match Python silence duration logic', () => {
        // Python: pre_silence = random.uniform(0.1, 0.5)
        // Python: post_silence = 0.55
        // JS: preSilence = 0.2 (Fixed)
        // JS: postSilence = 0.2 (Fixed) -> THIS IS THE DIFFERENCE
        
        // We need to verify if this difference causes the issue.
        // For now, let's just check if the JS implementation respects its own constants.
        
        const wpm = 20;
        const timing = gen.generateTiming('E', wpm);
        const waveform = gen.generateWaveform(timing);
        
        const dotLen = 1.2 / wpm;
        const expectedDuration = dotLen + 0.2 + 0.55; // timing + pre + post
        const expectedSamples = Math.floor(expectedDuration * sampleRate);
        
        expect(Math.abs(waveform.length - expectedSamples)).toBeLessThanOrEqual(1);
    });
});

describe('HFChannelSimulator', () => {
    const sampleRate = 16000;
    const sim = new HFChannelSimulator(sampleRate);

    it('should apply noise', () => {
        const waveform = new Float32Array(1000).fill(0);
        const noisy = sim.applyNoise(waveform, 10);
        expect(noisy.length).toBe(waveform.length);
        
        let hasNoise = false;
        for (let i = 0; i < noisy.length; i++) {
            if (noisy[i] !== 0) {
                hasNoise = true;
                break;
            }
        }
        expect(hasNoise).toBe(true);
    });

    it('should apply filter', () => {
        const waveform = new Float32Array(1000).fill(0.1);
        const filtered = sim.applyFilter(waveform, 700, 500);
        expect(filtered.length).toBe(waveform.length);
        expect(filtered).toBeInstanceOf(Float32Array);
    });

    it('should handle very short waveform', () => {
        const waveform = new Float32Array(10).fill(0.1);
        const noisy = sim.applyNoise(waveform, 10);
        const filtered = sim.applyFilter(waveform, 700, 500);
        expect(noisy.length).toBe(10);
        expect(filtered.length).toBe(10);
    });
});