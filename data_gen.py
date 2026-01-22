"""
CW Data Generator module.
Responsible for synthesizing Morse code signals with human keying artifacts
and HF channel simulations (noise, fading, QRM).
"""

import torch
import numpy as np
import scipy.signal
import scipy.io.wavfile
import random
import string
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
import config

# Morse Code Definition
MORSE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', '.': '.-.-.-', ',': '--..--', '?': '..--..',
    '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-', ' ': ' '
}

class MorseGenerator:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def text_to_morse(self, text: str) -> str:
        text = text.upper()
        return " ".join([MORSE_DICT.get(c, "") for c in text])

    def generate_timing(self, text: str, wpm: int = 20, farnsworth_wpm: int = None, 
                        jitter: float = 0.0, weight: float = 1.0) -> List[Tuple[bool, float]]:
        """
        Generate timing sequence (is_on, duration_sec) for the given text.
        """
        if farnsworth_wpm is None:
            farnsworth_wpm = wpm
        
        dot_len = 1.2 / wpm
        # Farnsworth timing: dots and dashes are at 'wpm', but spaces are at 'farnsworth_wpm'
        char_space_len = (3 * 1.2 / farnsworth_wpm)
        word_space_len = (7 * 1.2 / farnsworth_wpm)
        
        timing = []
        words = text.upper().split(' ')
        
        for i, word in enumerate(words):
            for j, char in enumerate(word):
                code = MORSE_DICT.get(char, "")
                for k, symbol in enumerate(code):
                    # Apply weight and jitter
                    if symbol == '.':
                        duration = dot_len * weight
                    else: # '-'
                        duration = dot_len * 3 * weight
                    
                    if jitter > 0:
                        duration *= (1 + random.uniform(-jitter, jitter))
                    
                    timing.append((True, duration))
                    
                    # Inter-symbol space (within a character)
                    if k < len(code) - 1:
                        timing.append((False, dot_len))
                
                # Inter-character space
                if j < len(word) - 1:
                    timing.append((False, char_space_len))
            
            # Inter-word space
            if i < len(words) - 1:
                timing.append((False, word_space_len))
        
        return timing

    def generate_waveform(self, timing: List[Tuple[bool, float]], frequency: float = 700.0, 
                          waveform_type: str = 'sine', rise_time: float = 0.005) -> np.ndarray:
        """
        Convert timing sequence to audio waveform.
        """
        total_duration = sum(t[1] for t in timing)
        total_samples = int(total_duration * self.sample_rate) + 100
        waveform = np.zeros(total_samples)
        
        current_sample = 0
        for is_on, duration in timing:
            num_samples = int(duration * self.sample_rate)
            if is_on:
                t = np.arange(num_samples) / self.sample_rate
                if waveform_type == 'sine':
                    sig = np.sin(2 * np.pi * frequency * t)
                elif waveform_type == 'square':
                    sig = scipy.signal.square(2 * np.pi * frequency * t)
                elif waveform_type == 'sawtooth':
                    sig = scipy.signal.sawtooth(2 * np.pi * frequency * t)
                else:
                    sig = np.sin(2 * np.pi * frequency * t)
                
                # Apply envelope (rise/fall) to avoid clicks
                envelope = np.ones(num_samples)
                n_rise = int(rise_time * self.sample_rate)
                if n_rise * 2 > num_samples:
                    n_rise = num_samples // 2
                
                if n_rise > 0:
                    rise = 0.5 * (1 - np.cos(np.pi * np.arange(n_rise) / n_rise))
                    envelope[:n_rise] = rise
                    envelope[-n_rise:] = rise[::-1]
                
                sig *= envelope
                end_sample = min(current_sample + num_samples, total_samples)
                waveform[current_sample:end_sample] = sig[:end_sample-current_sample]
            
            current_sample += num_samples
            
        return waveform

class HFChannelSimulator:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def apply_fading(self, waveform: np.ndarray, speed_hz: float = 0.1) -> np.ndarray:
        """Apply Rayleigh-like fading using filtered Gaussian noise."""
        # Generate complex Gaussian noise
        n_samples = len(waveform)
        # Low-pass filter to simulate fading speed (Doppler spread)
        nyquist = 0.5 * self.sample_rate
        b, a = scipy.signal.butter(2, speed_hz / nyquist, btype='low')
        
        # Real and imaginary parts for Rayleigh
        r_real = scipy.signal.lfilter(b, a, np.random.randn(n_samples))
        r_imag = scipy.signal.lfilter(b, a, np.random.randn(n_samples))
        
        # Rayleigh envelope
        fading = np.sqrt(r_real**2 + r_imag**2)
        
        # Normalize fading envelope
        fading /= (np.mean(fading) + 1e-12)
        # Avoid total silence
        fading = np.clip(fading, 0.05, 2.0)
        
        return waveform * fading

    def apply_noise(self, waveform: np.ndarray, snr_db: float = 10.0, impulse_prob: float = 0.001) -> np.ndarray:
        """Apply AWGN and impulse noise."""
        # AWGN
        sig_avg_watts = np.mean(waveform**2)
        sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-12)
        noise_avg_db = sig_avg_db - snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_avg_watts), len(waveform))
        
        # Impulse noise
        impulses = np.zeros(len(waveform))
        if impulse_prob > 0:
            n_impulses = int(len(waveform) * impulse_prob)
            indices = np.random.randint(0, len(waveform), n_impulses)
            impulses[indices] = np.random.uniform(-1, 1, n_impulses)
            
        return waveform + noise + impulses

    def apply_qrm(self, waveform: np.ndarray, snr_db: float = 5.0) -> np.ndarray:
        """Apply interference from another Morse signal."""
        # Simplified QRM: just another tone with some offset
        t = np.arange(len(waveform)) / self.sample_rate
        offset = random.uniform(-200, 200)
        if abs(offset) < 50: offset = 50 # Avoid exact match
        
        qrm_freq = 700 + offset
        qrm = np.sin(2 * np.pi * qrm_freq * t)
        
        # Simple on-off for QRM
        qrm_mask = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(float)
        qrm *= qrm_mask
        
        sig_avg_watts = np.mean(waveform**2)
        qrm_avg_watts = sig_avg_watts / (10 ** (snr_db / 10))
        qrm *= np.sqrt(qrm_avg_watts + 1e-12)
        
        return waveform + qrm

    def apply_filter(self, waveform: np.ndarray, center_freq: float = 700.0, bandwidth: float = 500.0) -> np.ndarray:
        """Apply Bandpass filter (Receiver characteristic)."""
        nyquist = 0.5 * self.sample_rate
        low = (center_freq - bandwidth / 2) / nyquist
        high = (center_freq + bandwidth / 2) / nyquist
        b, a = scipy.signal.butter(4, [max(0.01, low), min(0.99, high)], btype='band')
        return scipy.signal.lfilter(b, a, waveform)

def generate_sample(text: str, wpm: int = 20, snr_db: float = 10.0, sample_rate: int = config.SAMPLE_RATE) -> Tuple[torch.Tensor, str]:
    gen = MorseGenerator(sample_rate=sample_rate)
    sim = HFChannelSimulator(sample_rate=sample_rate)
    
    # Human artifacts
    jitter = random.uniform(0.01, 0.1)
    weight = random.uniform(0.8, 1.2)
    
    timing = gen.generate_timing(text, wpm=wpm, jitter=jitter, weight=weight)
    waveform = gen.generate_waveform(timing)
    
    waveform = sim.apply_fading(waveform)
    if random.random() > 0.5:
        waveform = sim.apply_qrm(waveform, snr_db=snr_db + random.uniform(0, 10))
    waveform = sim.apply_noise(waveform, snr_db=snr_db)
    waveform = sim.apply_filter(waveform)
    
    # Normalize
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
        
    return torch.from_numpy(waveform).float(), text

class CWDataset(Dataset):
    def __init__(self, num_samples: int = 1000, min_wpm: int = 10, max_wpm: int = 40,
                 min_snr: float = 5.0, max_snr: float = 25.0):
        self.num_samples = num_samples
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.chars = string.ascii_uppercase + string.digits + "/?.,"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random text
        length = random.randint(5, 20)
        text = "".join(random.choices(self.chars + " ", k=length)).strip()
        if not text: text = "CQ"
        
        wpm = random.randint(self.min_wpm, self.max_wpm)
        snr = random.uniform(self.min_snr, self.max_snr)
        
        waveform, label = generate_sample(text, wpm=wpm, snr_db=snr)
        return waveform, label

if __name__ == "__main__":
    sample_text = "CQ DE KILO CODE K"
    sample_rate = config.SAMPLE_RATE
    print(f"Generating sample: {sample_text}")
    
    waveform, label = generate_sample(sample_text, wpm=25, snr_db=15, sample_rate=sample_rate)
    
    output_file = "sample_cw.wav"
    # Convert back to numpy for scipy saving
    wf_np = waveform.numpy()
    scipy.io.wavfile.write(output_file, sample_rate, (wf_np * 32767).astype(np.int16))
    
    print(f"Saved to {output_file} using scipy")
    
    # Test Dataset
    dataset = CWDataset(num_samples=5)
    wf, lbl = dataset[0]
    print(f"Dataset test - Waveform shape: {wf.shape}, Label: {lbl}")