/**
 * CW Data Generator (Ported from data_gen.py)
 */
import { DSP } from './dsp.js';

export const MORSE_DICT = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    "\"": ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
    "<AA>": ".-.-",
    "<KA>": "-.-.-",
    "<SK>": "...-.-",
    "<VE>": "...-.",
    "<HH>": "........",
    "<NJ>": "-..---",
    "<SN>": "...-. ",
    "<SOS>": "...---...",
};

export class MorseGenerator {
    constructor(sampleRate = 16000) {
        this.sampleRate = sampleRate;
    }

    textToMorseTokens(text) {
        const tokens = [];
        let i = 0;
        while (i < text.length) {
            if (text[i] === '<') {
                const end = text.indexOf('>', i);
                if (end !== -1) {
                    const token = text.substring(i, end + 1);
                    if (MORSE_DICT[token]) {
                        tokens.push(token);
                        i = end + 1;
                        continue;
                    }
                }
            }
            if (text[i] !== ' ') {
                tokens.push(text[i].toUpperCase());
            }
            i++;
        }
        return tokens;
    }

    generateTiming(text, wpm = 20, farnsworthWpm = null, jitter = 0.0, weight = 1.0) {
        if (farnsworthWpm === null) {
            farnsworthWpm = wpm;
        }

        const dotLen = 1.2 / wpm;
        const charSpaceLen = (3 * 1.2 / farnsworthWpm);
        const wordSpaceLen = (7 * 1.2 / farnsworthWpm);

        const timing = [];
        const wordsRaw = text.split(' ');

        for (let i = 0; i < wordsRaw.length; i++) {
            const wordRaw = wordsRaw[i];
            if (!wordRaw) continue;
            const tokens = this.textToMorseTokens(wordRaw);

            for (let j = 0; j < tokens.length; j++) {
                const char = tokens[j];
                const code = MORSE_DICT[char] || "";
                for (let k = 0; k < code.length; k++) {
                    const symbol = code[k];
                    let duration = 0;
                    let classId = 0;

                    if (symbol === '.') {
                        duration = dotLen * weight;
                        classId = 1; // Dit
                    } else if (symbol === '-') {
                        duration = dotLen * 3 * weight;
                        classId = 2; // Dah
                    } else if (symbol === ' ') {
                        timing.push({ classId: 4, duration: charSpaceLen });
                        continue;
                    }

                    if (jitter > 0) {
                        duration *= (1 + (Math.random() * 2 - 1) * jitter);
                    }

                    if (classId > 0) {
                        timing.push({ classId, duration });
                    }

                    if (k < code.length - 1 && code[k + 1] !== ' ') {
                        timing.push({ classId: 3, duration: dotLen }); // Intra-char space
                    }
                }

                if (j < tokens.length - 1) {
                    timing.push({ classId: 4, duration: charSpaceLen }); // Inter-char space
                }
            }

            if (i < wordsRaw.length - 1) {
                timing.push({ classId: 5, duration: wordSpaceLen }); // Inter-word space
            }
        }

        return timing;
    }

    /**
     * Estimate the WPM needed to fit the text into targetFrames.
     * Ported from data_gen.py
     */
    estimateWpmForTargetFrames(text, targetFrames = 1000, minWpm = 10, maxWpm = 45) {
        const tokens = this.textToMorseTokens(text);
        let totalUnits = 0;
        for (const token of tokens) {
            const code = MORSE_DICT[token] || "";
            for (const symbol of code) {
                if (symbol === '.') totalUnits += 1;
                else if (symbol === '-') totalUnits += 3;
                totalUnits += 1; // Intra-char space
            }
            totalUnits += 3; // Inter-char space
        }

        const hopLength = 160; // config.HOP_LENGTH
        const targetSec = targetFrames * hopLength / this.sampleRate;

        if (targetSec <= 0) return maxWpm;
        const neededWpm = (1.2 * totalUnits) / targetSec;

        return Math.max(minWpm, Math.min(maxWpm, Math.round(neededWpm)));
    }

    generateWaveform(timing, frequency = 700.0, riseTime = 0.005) {
        const preSilence = 0.2; // Fixed for demo generation
        const postSilence = 0.55; // Sync with Python data_gen.py
        const totalDuration = timing.reduce((acc, t) => acc + t.duration, 0) + preSilence + postSilence;
        const totalSamples = Math.floor(totalDuration * this.sampleRate);
        const waveform = new Float32Array(totalSamples);

        let currentSample = Math.floor(preSilence * this.sampleRate);
        for (const { classId, duration } of timing) {
            const numSamples = Math.floor(duration * this.sampleRate);
            const isOn = (classId === 1 || classId === 2);

            if (isOn) {
                for (let i = 0; i < numSamples; i++) {
                    const t = i / this.sampleRate;
                    let sig = Math.sin(2 * Math.PI * frequency * t);

                    // Apply envelope
                    let envelope = 1.0;
                    const nRise = Math.floor(riseTime * this.sampleRate);
                    if (i < nRise) {
                        envelope = 0.5 * (1 - Math.cos(Math.PI * i / nRise));
                    } else if (i > numSamples - nRise) {
                        const j = i - (numSamples - nRise);
                        envelope = 0.5 * (1 + Math.cos(Math.PI * j / nRise));
                    }

                    if (currentSample + i < totalSamples) {
                        waveform[currentSample + i] = sig * envelope;
                    }
                }
            }
            currentSample += numSamples;
        }

        return waveform;
    }
}

export class HFChannelSimulator {
    constructor(sampleRate = 16000) {
        this.sampleRate = sampleRate;
    }

    applyNoise(waveform, snrDb = 10.0) {
        // SNR is defined based on the average power of the signal during the MARK (ON) state.
        // For a sine wave with amplitude 1.0, the power is 0.5.
        const markPower = 0.5;
        const noisePower = markPower / Math.pow(10, snrDb / 10);
        const sigma = Math.sqrt(noisePower);
        const noise = DSP.generateGaussianNoise(waveform.length, sigma);

        const result = new Float32Array(waveform.length);
        for (let i = 0; i < waveform.length; i++) {
            result[i] = waveform[i] + noise[i];
        }
        return result;
    }

    applyFilter(waveform, centerFreq = 700.0, bandwidth = 500.0) {
        // Match Python's scipy.signal.butter(4, ..., btype='band') which is 8th order
        // If parameters match default (700Hz, 500Hz BW, 16k SR), use pre-calculated coefficients
        if (Math.abs(centerFreq - 700.0) < 1e-6 && Math.abs(bandwidth - 500.0) < 1e-6 && this.sampleRate === 16000) {
            const b = [
                7.277254928998077e-05, 0.0, -0.0002910901971599231, 0.0, 0.0004366352957398846,
                0.0, -0.0002910901971599231, 0.0, 7.277254928998077e-05
            ];
            const a = [
                1.0, -7.241066256002014, 23.17663287780428, -42.81975359431594, 49.94114683253107,
                -37.65166894602891, 17.92036688875102, -4.9237085887453595, 0.5980652616008868
            ];
            return DSP.lfilter(b, a, waveform);
        }

        // Fallback to simpler filter for other frequencies (or implement full butter design)
        // Simple Biquad Bandpass Filter implementation
        const f0 = centerFreq / this.sampleRate;
        const Q = centerFreq / bandwidth;
        const omega = 2 * Math.PI * f0;
        const alpha = Math.sin(omega) / (2 * Q);

        const b = [alpha, 0, -alpha];
        const a = [1 + alpha, -2 * Math.cos(omega), 1 - alpha];

        return DSP.lfilter(b, a, waveform);
    }
}