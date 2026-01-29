/**
 * Shared visualization utilities for CW Decoder demos.
 */

// Accurate Viridis colormap points (9 control points from 0.0 to 1.0)
export const VIRIDIS = [
    [68, 1, 84],     // 0.0
    [72, 40, 120],   // 0.125
    [62, 74, 137],   // 0.25
    [49, 104, 142],  // 0.375
    [38, 130, 142],  // 0.5
    [31, 158, 137],  // 0.625
    [53, 183, 121],  // 0.75
    [109, 205, 89],  // 0.875
    [253, 231, 37]   // 1.0
];

/**
 * Get RGB color from Viridis colormap.
 * @param {number} v - Value between 0 and 1
 * @returns {number[]} RGB array [r, g, b]
 */
export function getViridisColor(v) {
    v = Math.max(0, Math.min(1, v));
    const pos = v * (VIRIDIS.length - 1);
    const idx = Math.floor(pos);
    const frac = pos - idx;
    if (idx >= VIRIDIS.length - 1) return VIRIDIS[VIRIDIS.length - 1];

    const c1 = VIRIDIS[idx];
    const c2 = VIRIDIS[idx + 1];
    return [
        Math.floor(c1[0] + (c2[0] - c1[0]) * frac),
        Math.floor(c1[1] + (c2[1] - c1[1]) * frac),
        Math.floor(c1[2] + (c2[2] - c1[2]) * frac)
    ];
}

/**
 * Convert power values to dB scale with adaptive normalization.
 * @param {Float32Array[]} frames - Array of spectrogram frames
 * @returns {{dbFrames: number[][], minDB: number, maxDB: number}}
 */
export function powerToDBNormalized(frames) {
    let minDB = Infinity;
    let maxDB = -Infinity;

    const dbFrames = frames.map(frame => {
        return Array.from(frame).map(p => {
            const db = 10 * Math.log10(p + 1e-9);
            if (db < minDB) minDB = db;
            if (db > maxDB) maxDB = db;
            return db;
        });
    });

    // Ensure a minimum range to avoid division by zero
    if (maxDB <= minDB) maxDB = minDB + 1;

    return { dbFrames, minDB, maxDB };
}

/**
 * Normalize dB value to 0-1 range.
 * @param {number} db - dB value
 * @param {number} minDB - Minimum dB
 * @param {number} maxDB - Maximum dB
 * @returns {number} Normalized value 0-1
 */
export function normalizeDB(db, minDB, maxDB) {
    return (db - minDB) / (maxDB - minDB);
}
