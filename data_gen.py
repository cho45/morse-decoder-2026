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
    '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-', ' ': ' ',
    '<BT>': '-...-', '<AR>': '.-.-.', '<SK>': '...-.-', '<KA>': '-.-.-',
    'CQ': '-.-. --.-', 'DE': '-.. .'
}

class MorseGenerator:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def text_to_morse(self, text: str) -> str:
        text = text.upper()
        return " ".join([MORSE_DICT.get(c, "") for c in text])

    def text_to_morse_tokens(self, text: str) -> List[str]:
        """Split text into tokens (chars and prosigns)."""
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == '<':
                end = text.find('>', i)
                if end != -1:
                    token = text[i:end+1]
                    if token in MORSE_DICT:
                        tokens.append(token)
                        i = end + 1
                        continue
            
            # Check for multi-char tokens like CQ, DE if they are treated as single tokens in config
            found = False
            for token in config.PROSIGNS:
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    found = True
                    break
            if found: continue

            tokens.append(text[i])
            i += 1
        return tokens

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
        # We need to handle prosigns which might contain spaces (like CQ) but are treated as one unit here?
        # No, MORSE_DICT['CQ'] = '-.-. --.-' which has space.
        # But our generate_timing logic splits by space for words.
        
        # Simpler approach: Pre-process text to handle tokens.
        # But wait, 'text' input here is the ground truth string.
        # If text contains "<BT>", we should treat it as one character.
        
        # Let's assume input text uses spaces for word boundaries.
        # We need to tokenize the text first.
        
        # Split by space to get words
        words_raw = text.split(' ')
        
        for i, word_raw in enumerate(words_raw):
            if not word_raw: continue
            
            # Tokenize word into chars/prosigns
            tokens = self.text_to_morse_tokens(word_raw)
            
            for j, char in enumerate(tokens):
                code = MORSE_DICT.get(char, "")
                # Special handling for tokens that map to sequence with spaces (like CQ)
                # Actually, standard prosigns don't have spaces inside (they are run together).
                # CQ and DE are abbreviations, so they are just "C Q" and "D E".
                # But if we treat "CQ" as a single token class, we should generate it as such.
                
                for k, symbol in enumerate(code):
                    # Apply weight and jitter
                    if symbol == '.':
                        duration = dot_len * weight
                        timing.append((True, duration))
                    elif symbol == '-':
                        duration = dot_len * 3 * weight
                        timing.append((True, duration))
                    elif symbol == ' ':
                        # Handle spaces within prosigns/tokens
                        timing.append((False, char_space_len))
                        continue # Skip the inter-symbol space below
                    
                    if jitter > 0:
                        timing[-1] = (timing[-1][0], timing[-1][1] * (1 + random.uniform(-jitter, jitter)))
                    
                    # Inter-symbol space (within a character)
                    if k < len(code) - 1 and code[k+1] != ' ':
                        timing.append((False, dot_len))
                
                # Inter-character space
                if j < len(tokens) - 1:
                    timing.append((False, char_space_len))
            
            # Inter-word space
            if i < len(words_raw) - 1:
                timing.append((False, word_space_len))
        
        return timing

    def generate_waveform(self, timing: List[Tuple[bool, float]], frequency: float = 700.0, 
                          waveform_type: str = 'sine', rise_time: float = 0.005) -> np.ndarray:
        """
        Convert timing sequence to audio waveform.
        """
        # Randomize pre_silence to improve alignment robustness.
        # This gives the causal Conformer varying history to see the start of a signal.
        pre_silence = random.uniform(0.1, 0.5)
        post_silence = 0.55
        total_duration = sum(t[1] for t in timing) + pre_silence + post_silence
        total_samples = int(total_duration * self.sample_rate)
        waveform = np.zeros(total_samples)
        
        current_sample = int(pre_silence * self.sample_rate)
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

def generate_sample(text: str, wpm: int = 20, snr_db: float = 10.0, sample_rate: int = config.SAMPLE_RATE,
                    jitter: float = 0.0, weight: float = 1.0) -> Tuple[torch.Tensor, str]:
    gen = MorseGenerator(sample_rate=sample_rate)
    sim = HFChannelSimulator(sample_rate=sample_rate)
    
    # Human artifacts are now controlled by arguments
    
    timing = gen.generate_timing(text, wpm=wpm, jitter=jitter, weight=weight)
    waveform = gen.generate_waveform(timing)
    
    # Only apply channel effects if SNR is below a certain threshold (e.g., 45dB)
    if snr_db < 45:
        waveform = sim.apply_fading(waveform)
        if random.random() > 0.5:
            waveform = sim.apply_qrm(waveform, snr_db=snr_db + random.uniform(0, 10))
        waveform = sim.apply_noise(waveform, snr_db=snr_db)
        waveform = sim.apply_filter(waveform)
    else:
        # Truly clean: no noise, no filter, no fading
        pass
    
    # Normalize
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
        
    return torch.from_numpy(waveform).float(), text

class CWDataset(Dataset):
    def __init__(self, num_samples: int = 1000, min_wpm: int = 10, max_wpm: int = 40,
                 min_snr: float = 5.0, max_snr: float = 25.0,
                 jitter_max: float = 0.1, weight_var: float = 0.2,
                 allowed_chars: str = None, min_len: int = 5, max_len: int = 10):
        self.num_samples = num_samples
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.jitter_max = jitter_max
        self.weight_var = weight_var
        self.min_len = min_len
        self.max_len = max_len
        # config.CHARS includes prosigns now
        self.all_chars = config.CHARS
        self.chars = allowed_chars if allowed_chars else self.all_chars

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random text
        length = random.randint(self.min_len, self.max_len)
        
        # Randomly choose from available tokens (chars + prosigns)
        # Filter out spaces for random choice, we will add them manually
        valid_tokens = [c for c in self.chars if c != ' ']
        
        # Create a list of tokens
        tokens = random.choices(valid_tokens, k=length)
        
        # Join them, occasionally adding spaces
        text = ""
        for t in tokens:
            text += t
            if random.random() < 0.2:
                text += " "
        text = text.strip()
        
        if not text: text = "CQ"
        
        wpm = random.randint(self.min_wpm, self.max_wpm)
        snr = random.uniform(self.min_snr, self.max_snr)
        
        # Determine jitter and weight based on curriculum settings
        jitter = random.uniform(0, self.jitter_max)
        # weight is centered at 1.0, variation is +/- weight_var
        weight = 1.0 + random.uniform(-self.weight_var, self.weight_var)
        
        waveform, label = generate_sample(text, wpm=wpm, snr_db=snr, jitter=jitter, weight=weight)
        # Return wpm as well so the trainer can use it for adaptive space reconstruction
        return waveform, label, wpm

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