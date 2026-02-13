/**
 * PeakDetector for finding stable carrier frequencies in CW signals.
 * Uses long-term temporal integration to isolate carriers and parabolic interpolation for sub-bin precision.
 */
export class PeakDetector {
    constructor(nBins) {
        if (!Number.isInteger(nBins) || nBins <= 0) {
            throw new Error(`Invalid nBins: ${nBins}`);
        }
        this.nBins = nBins;
        this.integratedSpectrum = new Float32Array(nBins);
        this.noiseFloor = 0.0;

        // Time constants: Attack (tau=500ms), Decay (tau=5s)
        // alpha ~= 1 - exp(-T / tau)
        // Assuming ~20ms per frame, 
        // alphaAttack ~= 0.04 -> use 0.05
        // alphaDecay ~= 0.004
        this.alphaAttack = 0.05;
        this.alphaDecay = 0.004;
        this.snrThreshold = 2.0;
    }

    /**
     * Process a new spectral magnitude frame.
     * @param {Float32Array} magnitudes 
     * @returns {Array} List of detected peaks {index, magnitude, snr} sorted by magnitude descending.
     */
    process(magnitudes) {
        if (magnitudes.length !== this.nBins) {
            throw new Error(`Expected ${this.nBins} bins, got ${magnitudes.length}`);
        }

        // 1. Temporal Integration (Asymmetric EMA)
        for (let i = 0; i < this.nBins; i++) {
            const mag = magnitudes[i];
            const current = this.integratedSpectrum[i];
            const alpha = mag > current ? this.alphaAttack : this.alphaDecay;
            this.integratedSpectrum[i] = alpha * mag + (1 - alpha) * current;
        }

        // 2. Noise Floor Estimation (Median of integrated spectrum)
        const sorted = new Float32Array(this.integratedSpectrum).sort();
        this.noiseFloor = sorted[Math.floor(this.nBins / 2)];

        // 3. Peak Extraction
        const peaks = [];
        const threshold = this.noiseFloor * this.snrThreshold;

        // Skip boundaries (0 and nBins-1) for parabolic interpolation
        for (let i = 1; i < this.nBins - 1; i++) {
            const y2 = this.integratedSpectrum[i];

            // Basic threshold and local maxima check
            if (y2 > threshold && y2 > this.integratedSpectrum[i - 1] && y2 > this.integratedSpectrum[i + 1]) {
                const y1 = this.integratedSpectrum[i - 1];
                const y3 = this.integratedSpectrum[i + 1];

                // 4. Parabolic Interpolation
                // delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                const denom = y1 - 2 * y2 + y3;
                let delta = 0;
                if (Math.abs(denom) > 1e-15) {
                    delta = 0.5 * (y1 - y3) / denom;
                }

                peaks.push({
                    index: i + delta,
                    magnitude: y2,
                    snr: this.noiseFloor > 0 ? y2 / this.noiseFloor : Infinity
                });
            }
        }

        // Sort by magnitude descending
        peaks.sort((a, b) => b.magnitude - a.magnitude);

        return peaks;
    }

    reset() {
        this.integratedSpectrum.fill(0);
        this.noiseFloor = 0.0;
    }
}
