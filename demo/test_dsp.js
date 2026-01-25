const DSP = require('./dsp.js');

function testFFT() {
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
    console.log(`FFT Test - Expected: ${freq}Hz, Detected: ${detectedFreq}Hz`);
    if (Math.abs(detectedFreq - freq) < (sr / n)) {
        console.log("FFT Test Passed!");
    } else {
        console.error("FFT Test Failed!");
        process.exit(1);
    }
}

testFFT();