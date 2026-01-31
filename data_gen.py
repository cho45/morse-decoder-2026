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
	"A":".-",
	"B":"-...",
	"C":"-.-.",
	"D":"-..",
	"E":".",
	"F":"..-.",
	"G":"--.",
	"H":"....",
	"I":"..",
	"J":".---",
	"K":"-.-",
	"L":".-..",
	"M":"--",
	"N":"-.",
	"O":"---",
	"P":".--.",
	"Q":"--.-",
	"R":".-.",
	"S":"...",
	"T":"-",
	"U":"..-",
	"V":"...-",
	"W":".--",
	"X":"-..-",
	"Y":"-.--",
	"Z":"--..",
	"0":"-----",
	"1":".----",
	"2":"..---",
	"3":"...--",
	"4":"....-",
	"5":".....",
	"6":"-....",
	"7":"--...",
	"8":"---..",
	"9":"----.",
	".":".-.-.-",
	",":"--..--",
	"?":"..--..",
	"'":".----.",
	"!":"-.-.--",
	"/":"-..-.",
	"(":"-.--.",
	")":"-.--.-",
	"&":".-...", # AS
	":":"---...",
	";":"-.-.-.",
	"=":"-...-", # BT (new paragraph)
	"+":".-.-.",  # AR (end of message)
	"-":"-....-",
	"_":"..--.-",
	"\"":".-..-.",
	"$":"...-..-",
	"@":".--.-.",
	"<AA>" : ".-.-", # AA (new line)
    "<KA>" : "-.-.-", # CT/KA (attention)
    "<SK>" : "...-.-", # VA/SK (end of transmission)
    '<VE>': '...-.',
    '<HH>': '........',
    '<NJ>': '-..---',
    '<DDD>': '-..-..-..',
    '<SOS>': '...---...',
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

            # Space is now a valid token in config.CHARS
            tokens.append(text[i])
            i += 1
        return tokens

    def generate_timing(self, text: str, wpm: int = 20, farnsworth_wpm: int = None,
                        jitter: float = 0.0, weight: float = 1.0) -> List[Tuple[int, float]]:
        """
        Generate timing sequence (class_id, duration_sec) for the given text.
        Classes: 1: Dit, 2: Dah, 3: Intra-char space, 4: Inter-char space, 5: Inter-word space
        """
        if farnsworth_wpm is None:
            farnsworth_wpm = wpm
        
        dot_len = 1.2 / wpm
        char_space_len = (3 * 1.2 / farnsworth_wpm)
        word_space_len = (7 * 1.2 / farnsworth_wpm)
        
        timing = []
        words_raw = text.split(' ')
        
        for i, word_raw in enumerate(words_raw):
            # i < len(words_raw) - 1 check handles the case where text ends with a space
            if not word_raw and i < len(words_raw) - 1: continue
            tokens = self.text_to_morse_tokens(word_raw)
            
            for j, char in enumerate(tokens):
                if char == ' ':
                    # Explicitly handle space tokens as Inter-word space (7 units)
                    timing.append((5, word_space_len))
                    continue
                code = MORSE_DICT.get(char, "")
                # print(f"DEBUG: token='{char}', code='{code}'")
                for k, symbol in enumerate(code):
                    if symbol == '.':
                        duration = dot_len * weight
                        timing.append((1, duration)) # Dit
                    elif symbol == '-':
                        duration = dot_len * 3 * weight
                        timing.append((2, duration)) # Dah
                    elif symbol == ' ':
                        timing.append((4, char_space_len)) # Inter-char space (for CQ etc)
                        continue
                    
                    if jitter > 0:
                        timing[-1] = (timing[-1][0], timing[-1][1] * (1 + random.uniform(-jitter, jitter)))
                    
                    if k < len(code) - 1 and code[k+1] != ' ':
                        timing.append((3, dot_len)) # Intra-char space
                
                if j < len(tokens) - 1:
                    timing.append((4, char_space_len)) # Inter-char space
            
            if i < len(words_raw) - 1:
                timing.append((5, word_space_len)) # Inter-word space
        
        return timing

    def estimate_wpm_for_target_frames(self, text: str, target_frames: int = config.TARGET_FRAMES,
                                      min_wpm: int = 10, max_wpm: int = 45) -> int:
        """
        Estimate the WPM needed to fit the text into target_frames.
        """
        # Roughly 50 units per word (PARIS standard)
        # 1 WPM = 50 units per minute = 50/60 units per second
        # Duration (sec) = units / (WPM * 50 / 60) = (1.2 * units) / WPM
        
        # Count approximate units in text
        tokens = self.text_to_morse_tokens(text)
        total_units = 0
        for token in tokens:
            code = MORSE_DICT.get(token, "")
            for symbol in code:
                if symbol == '.': total_units += 1
                elif symbol == '-': total_units += 3
                total_units += 1 # Intra-char space
            total_units += 3 # Inter-char space
        
        target_sec = target_frames * config.HOP_LENGTH / self.sample_rate
        
        # WPM = (1.2 * units) / sec
        if target_sec <= 0: return max_wpm
        needed_wpm = (1.2 * total_units) / target_sec
        
        return int(np.clip(needed_wpm, min_wpm, max_wpm))

    def estimate_max_chars_for_wpm(self, wpm: int, target_frames: int = config.TARGET_FRAMES) -> int:
        """
        Estimate the maximum number of characters that can fit in target_frames at given WPM.
        Using PARIS standard (50 units per word, 5 chars + 1 space).
        """
        # Subtract silence (approx 1.0s total for pre/post silence)
        target_sec = max(0.5, (target_frames * config.HOP_LENGTH / self.sample_rate) - 1.0)
        
        # total_units = target_sec * (WPM * 50 / 60) = target_sec * WPM / 1.2
        total_units = (target_sec * wpm) / 1.2
        
        # Average units per character:
        # PARIS is 50 units for 5 chars + 1 space = 50/6 = 8.33 units/char.
        # Some characters are long (e.g., '0' is 19 units).
        # We use a very conservative 15 units/char to ensure it fits even with numbers.
        max_chars = int(total_units / 15)
        return max(1, max_chars)

    def generate_waveform(self, timing: List[Tuple[int, float]], frequency: float = 700.0,
                          waveform_type: str = 'sine', rise_time: float = 0.005, wpm: int = 20,
                          drift_hz: float = 0.0, max_duration: float = 10.0) -> np.ndarray:
        """
        Convert timing sequence to audio waveform.
        Always returns a waveform of exactly max_duration seconds.
        """
        total_timing_duration = sum(t[1] for t in timing)
        
        # Ensure we fit in max_duration
        if total_timing_duration > max_duration - 0.2:
            # This should have been handled by caller, but we truncate just in case
            pass

        # Randomly place the signal within the 10s window
        # Allow at least 0.1s at the start and end
        max_start = max(0.1, max_duration - total_timing_duration - 0.1)
        pre_silence = random.uniform(0.1, max_start)
        
        total_samples = int(max_duration * self.sample_rate)
        waveform = np.zeros(total_samples)
        
        # Frequency drift phase accumulation
        phase = 0.0
        
        current_sample = int(pre_silence * self.sample_rate)
        for class_id, duration in timing:
            num_samples = int(duration * self.sample_rate)
            is_on = class_id in [1, 2]
            if is_on:
                t = np.arange(num_samples) / self.sample_rate
                # Apply drift by modulating the instantaneous frequency
                if drift_hz > 0:
                    # Slow sinusoidal drift
                    inst_freq = frequency + drift_hz * np.sin(2 * np.pi * 0.2 * (current_sample + np.arange(num_samples)) / self.sample_rate)
                else:
                    inst_freq = np.full(num_samples, frequency)
                
                # Update phase based on instantaneous frequency
                d_phase = 2 * np.pi * inst_freq / self.sample_rate
                sig_phase = phase + np.cumsum(d_phase)
                phase = sig_phase[-1] % (2 * np.pi)
                
                if waveform_type == 'sine':
                    sig = np.sin(sig_phase)
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
                if current_sample < total_samples:
                    waveform[current_sample:end_sample] = sig[:end_sample-current_sample]
            
            current_sample += num_samples
            
        # Generate multi-class frame-level labels
        # 0: Background/Space, 1: Dit, 2: Dah, 3: Inter-word space
        num_frames = (total_samples - config.N_FFT) // config.HOP_LENGTH + 1
        signal_frames = np.zeros(num_frames, dtype=np.int64) # 0: Background
        boundary_frames = np.zeros(num_frames, dtype=np.float32)
        
        # 境界（Boundary）の定義:
        # 文字間空白(3ユニット)または単語間空白(7ユニット)が完了した瞬間のフレーム。
        # つまり、次の文字が開始される直前。
        
        # 境界ラベルを確実に立てるためのロジック
        dot_len_sec = 1.2 / wpm
        time_ptr = int(pre_silence * self.sample_rate)
        for class_id, duration in timing:
            duration_samples = int(duration * self.sample_rate)
            if class_id == 4: # Inter-char space (文字の終了)
                # 空白の終了時点（＝次の要素の開始直前）を特定
                trigger_sample = time_ptr + duration_samples
                trigger_frame = trigger_sample // config.HOP_LENGTH
                for offset in range(5):
                    if 0 <= trigger_frame + offset < num_frames:
                        boundary_frames[trigger_frame + offset] = 1.0
            elif class_id == 5: # Inter-word space (単語間空白 = 前の文字の終了 + スペース文字の終了)
                # 1. 前の文字の終了境界 (空白開始から 3ユニット後)
                # 単語間空白(7ユニット)のうち、最初の3ユニットを文字間空白、残り4ユニットをスペース文字分とみなす
                char_end_trigger = time_ptr + int(3 * dot_len_sec * self.sample_rate)
                char_end_frame = char_end_trigger // config.HOP_LENGTH
                for offset in range(5):
                    if 0 <= char_end_frame + offset < num_frames:
                        boundary_frames[char_end_frame + offset] = 1.0
                
                # 2. スペース文字自体の終了境界 (空白の終了時点)
                space_end_trigger = time_ptr + duration_samples
                space_end_frame = space_end_trigger // config.HOP_LENGTH
                for offset in range(5):
                    if 0 <= space_end_frame + offset < num_frames:
                        boundary_frames[space_end_frame + offset] = 1.0
            
            time_ptr += duration_samples

        # すべての文字（スペースを含む）の終了時に境界が立つようになったため、
        # ここでの末尾の自動生成は不要。

        for i in range(num_frames):
            center_sample = i * config.HOP_LENGTH + config.N_FFT // 2
            time_ptr = int(pre_silence * self.sample_rate)
            for class_id, duration in timing:
                duration_samples = int(duration * self.sample_rate)
                if time_ptr <= center_sample < time_ptr + duration_samples:
                    # Simplify classes:
                    # Original: 1: Dit, 2: Dah, 3: Intra, 4: Inter-char, 5: Inter-word
                    # New: 1: Dit, 2: Dah, 0: Space, 3: Inter-word
                    if class_id in [1, 2]:
                        signal_frames[i] = class_id
                    elif class_id == 5:
                        signal_frames[i] = 3
                    else:
                        signal_frames[i] = 0
                    break
                time_ptr += duration_samples
                if time_ptr > center_sample:
                    break
                    
        return waveform, signal_frames, boundary_frames

class HFChannelSimulator:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def apply_fading(self, waveform: np.ndarray, speed_hz: float = 0.1, min_fading: float = 0.05) -> np.ndarray:
        """Apply Rayleigh-like fading using filtered Gaussian noise."""
        if speed_hz <= 0:
            return waveform
            
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
        fading = np.clip(fading, min_fading, 2.0)
        
        return waveform * fading

    def apply_noise(self, waveform: np.ndarray, impulse_prob: float = 0.0, snr_2500: float = 10.0) -> np.ndarray:
        """Apply AWGN and impulse noise."""

        # AWGN
        # SNR is defined based on the average power of the signal during the MARK (ON) state.
        # For a sine wave with amplitude 1.0, the power is 0.5.
        mark_power = 0.5
        
        # C/N0 (dB-Hz) calculation
        # SNR_2500 = C/N0 - 10*log10(2500)
        cn0_db_hz = snr_2500 + 10 * np.log10(config.SNR_REF_BW)
        
        # Noise power density N0 (Watts/Hz)
        n0 = mark_power / (10**(cn0_db_hz / 10))
        
        # Total noise power in the full bandwidth (Fs/2)
        noise_power = n0 * (self.sample_rate / 2)
        
        noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
        
        # Impulse noise
        impulses = np.zeros(len(waveform))
        if impulse_prob > 0:
            n_impulses = int(len(waveform) * impulse_prob)
            indices = np.random.randint(0, len(waveform), n_impulses)
            impulses[indices] = np.random.uniform(-1, 1, n_impulses)
            
        return waveform + noise + impulses

    def apply_qrm(self, waveform: np.ndarray, snr_2500: float = 5.0) -> np.ndarray:
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
        # Use same bandwidth normalization as AWGN for QRM
        cn0_db_hz = snr_2500 + 10 * np.log10(config.SNR_REF_BW)
        n0 = sig_avg_watts / (10**(cn0_db_hz / 10))
        qrm_avg_watts = n0 * (self.sample_rate / 2)
        
        qrm *= np.sqrt(qrm_avg_watts + 1e-12)
        
        return waveform + qrm

    def apply_qrn(self, waveform: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply bursty static crashes (QRN)."""
        n_samples = len(waveform)
        qrn = np.zeros(n_samples)
        # Randomly place 1-5 crashes
        for _ in range(random.randint(1, 5)):
            duration = int(self.sample_rate * random.uniform(0.01, 0.05))
            start = random.randint(0, max(0, n_samples - duration))
            # Bursty noise: white noise multiplied by a window
            burst = np.random.normal(0, strength, duration)
            window = scipy.signal.windows.hann(duration)
            qrn[start:start+duration] += burst * window
        return waveform + qrn

    def apply_out_of_band_qrm(self, waveform: np.ndarray, target_freq: float, strength: float = 2.0) -> np.ndarray:
        """Apply strong signal outside the target filter band to trigger AGC."""
        t = np.arange(len(waveform)) / self.sample_rate
        # Offset significantly from target (e.g., 1-2 kHz away)
        offset = random.choice([-1500, 1500]) + random.uniform(-200, 200)
        qrm_freq = target_freq + offset
        qrm = np.sin(2 * np.pi * qrm_freq * t) * strength
        # Simple on-off pattern
        qrm *= (np.sin(2 * np.pi * 0.3 * t) > 0).astype(float)
        return waveform + qrm

    def apply_agc(self, waveform: np.ndarray, attack_ms: float = 5.0, release_ms: float = 500.0,
                  target_lvl: float = 0.5) -> np.ndarray:
        """
        Simulate Automatic Gain Control (AGC).
        Strong signals/noise reduce gain, which recovers slowly.
        """
        n_samples = len(waveform)
        gain = np.ones(n_samples)
        current_gain = 1.0
        
        # Time constants in samples
        alpha_attack = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000.0))
        alpha_release = np.exp(-1.0 / (release_ms * self.sample_rate / 1000.0))
        
        # Simple envelope follower
        envelope = 0.0
        for i in range(n_samples):
            abs_val = abs(waveform[i])
            if abs_val > envelope:
                envelope = alpha_attack * envelope + (1 - alpha_attack) * abs_val
            else:
                envelope = alpha_release * envelope + (1 - alpha_release) * abs_val
            
            # Gain is inversely proportional to envelope if above target
            if envelope > target_lvl:
                desired_gain = target_lvl / (envelope + 1e-6)
            else:
                desired_gain = 1.0
            
            # Smooth gain changes
            if desired_gain < current_gain: # Attack
                current_gain = alpha_attack * current_gain + (1 - alpha_attack) * desired_gain
            else: # Release
                current_gain = alpha_release * current_gain + (1 - alpha_release) * desired_gain
            
            gain[i] = current_gain
            
        return waveform * gain

    def apply_frequency_drift(self, waveform: np.ndarray, drift_hz: float = 10.0) -> np.ndarray:
        """Apply slow frequency drift using phase modulation."""
        t = np.arange(len(waveform)) / self.sample_rate
        # Slow drift (0.2 Hz modulation)
        drift = drift_hz * np.sin(2 * np.pi * 0.2 * t)
        # Phase is integral of frequency
        phase_drift = 2 * np.pi * np.cumsum(drift) / self.sample_rate
        
        # This is tricky because we only have the mixed waveform.
        # For a pure sine wave, we could just add phase.
        # For a complex signal, we approximate using a Hilbert transform for SSB-like shift,
        # but for simplicity and speed, we'll only apply this if we had the raw signal.
        # Since we mix later, we should move drift to MorseGenerator.generate_waveform.
        return waveform

    def apply_multipath(self, waveform: np.ndarray, delay_ms: float = 20.0, attenuation: float = 0.5) -> np.ndarray:
        """Apply simple multipath (echo)."""
        delay_samples = int(delay_ms * self.sample_rate / 1000.0)
        if delay_samples >= len(waveform):
            return waveform
        echo = np.zeros_like(waveform)
        echo[delay_samples:] = waveform[:-delay_samples] * attenuation
        return waveform + echo

    def apply_clipping(self, waveform: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Apply non-linear distortion (clipping)."""
        return np.clip(waveform, -threshold, threshold)

    def apply_filter(self, waveform: np.ndarray, center_freq: float = 700.0, bandwidth: float = 500.0) -> np.ndarray:
        """Apply Bandpass filter (Receiver characteristic)."""
        nyquist = 0.5 * self.sample_rate
        low = (center_freq - bandwidth / 2) / nyquist
        high = (center_freq + bandwidth / 2) / nyquist
        b, a = scipy.signal.butter(4, [max(0.01, low), min(0.99, high)], btype='band')
        return scipy.signal.lfilter(b, a, waveform)

    def apply_tx_filter(self, waveform: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply Low-pass filter to simulate TX-side bandwidth limitation (soften edges)."""
        nyquist = 0.5 * self.sample_rate
        # Ensure cutoff is within a reasonable range
        cutoff = np.clip(cutoff, 100, nyquist - 100)
        # Use higher order for more noticeable effect
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='low')
        return scipy.signal.lfilter(b, a, waveform)

def generate_sample(text: str, wpm: int = 20, sample_rate: int = config.SAMPLE_RATE,
                    jitter: float = 0.0, weight: float = 1.0,
                    fading_speed: float = 0.1, min_fading: float = 0.05,
                    frequency: float = 700.0,
                    tx_lowpass: float = None,
                    rise_time: float = 0.005,
                    min_gain_db: float = 0.0,
                    drift_hz: float = 0.0,
                    qrn_strength: float = 0.0,
                    qrm_prob: float = 0.1,
                    impulse_prob: float = 0.001,
                    agc_enabled: bool = False,
                    multipath_delay: float = 0.0,
                    clipping_threshold: float = 1.0,
                    max_duration: float = 10.0,
                    snr_2500: float = 10.0) -> Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:

    gen = MorseGenerator(sample_rate=sample_rate)
    sim = HFChannelSimulator(sample_rate=sample_rate)
    
    # Human artifacts are now controlled by arguments
    
    # ワードの最後にも必ずスペースが入るようにし、境界ラベルの挙動を一貫させる
    if not text.endswith(" "):
        text += " "
        
    timing = gen.generate_timing(text, wpm=wpm, jitter=jitter, weight=weight)
    waveform, signal_labels, boundary_labels = gen.generate_waveform(
        timing, frequency=frequency, wpm=wpm, rise_time=rise_time, drift_hz=drift_hz, max_duration=max_duration
    )
    
    # Apply TX filter (soften edges) before channel effects
    if tx_lowpass is not None:
        waveform = sim.apply_tx_filter(waveform, cutoff=tx_lowpass)

    # Only apply channel effects if SNR is below a certain threshold (e.g., 45dB)
    if snr_2500 < 45:
        # 1. Channel propagation effects
        waveform = sim.apply_fading(waveform, speed_hz=fading_speed, min_fading=min_fading)
        if multipath_delay > 0:
            waveform = sim.apply_multipath(waveform, delay_ms=multipath_delay)

        # 2. Add noise and interference (Antenna input)
        waveform = sim.apply_noise(waveform, snr_2500=snr_2500, impulse_prob=impulse_prob)
        if random.random() < qrm_prob:
            waveform = sim.apply_qrm(waveform, snr_2500=snr_2500 + random.uniform(0, 10))
        if qrn_strength > 0:
            waveform = sim.apply_qrn(waveform, strength=qrn_strength)
        if agc_enabled and random.random() < qrm_prob:
            waveform = sim.apply_out_of_band_qrm(waveform, target_freq=frequency, strength=random.uniform(2.0, 5.0))

        # 3. Receiver stage 1: Filtering (Bandpass)
        waveform = sim.apply_filter(waveform, center_freq=frequency)
        
        # 4. Receiver stage 2: AGC (Reacts to filtered signal + remaining noise)
        if agc_enabled:
            waveform = sim.apply_agc(waveform)
            
        # 5. Receiver stage 3: Clipping (Final saturation)
        if clipping_threshold < 1.0:
            waveform = sim.apply_clipping(waveform, threshold=clipping_threshold)
    else:
        # Truly clean: no noise, no filter, no fading
        pass
    
    # Normalize and apply random gain augmentation
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
        # Apply random gain in dB scale if min_gain_db < 0.0
        if min_gain_db < 0.0:
            gain_db = random.uniform(min_gain_db, 0.0)
            gain = 10 ** (gain_db / 20)
            waveform *= gain
        
    return torch.from_numpy(waveform).float(), text, torch.from_numpy(signal_labels).float(), torch.from_numpy(boundary_labels).float()

class CWDataset(Dataset):
    def __init__(self, num_samples: int = 1000, min_wpm: int = 15, max_wpm: int = 40,
                 min_snr_2500: float = 10.0, max_snr_2500: float = 30.0,
                 jitter_max: float = 0.1, weight_var: float = 0.2,
                 allowed_chars: str = None, min_len: int = 5, max_len: int = 10,
                 focus_chars: str = None, focus_prob: float = 0.5,
                 fading_speed_min: float = 0.1, fading_speed_max: float = 0.1,
                 min_fading: float = 0.1,
                 phrase_prob: float = 0.0,
                 min_freq: float = 650.0,
                 max_freq: float = 750.0,
                 tx_lowpass_prob: float = 0.8,
                 rise_time_max: float = 0.025,
                 min_gain_db: float = 0.0,
                 drift_prob: float = 0.0,
                 qrn_prob: float = 0.0,
                 qrm_prob: float = 0.1,
                 impulse_prob: float = 0.001,
                 agc_prob: float = 0.0,
                 multipath_prob: float = 0.0,
                 clipping_prob: float = 0.0):

        self.num_samples = num_samples
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm
        self.min_snr_2500 = min_snr_2500
        self.max_snr_2500 = max_snr_2500
        self.jitter_max = jitter_max
        self.weight_var = weight_var
        self.min_len = min_len
        self.max_len = max_len
        self.fading_speed_min = fading_speed_min
        self.fading_speed_max = fading_speed_max
        self.min_fading = min_fading
        # config.CHARS includes prosigns now
        self.all_chars = config.CHARS
        self.chars = allowed_chars if allowed_chars else self.all_chars
        self.focus_chars = focus_chars
        self.focus_prob = focus_prob
        self.phrase_prob = phrase_prob
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.tx_lowpass_prob = tx_lowpass_prob
        self.rise_time_max = rise_time_max
        self.min_gain_db = min_gain_db
        self.drift_prob = drift_prob
        self.qrn_prob = qrn_prob
        self.qrm_prob = qrm_prob
        self.impulse_prob = impulse_prob
        self.agc_prob = agc_prob
        self.multipath_prob = multipath_prob
        self.clipping_prob = clipping_prob
        self.gen = MorseGenerator()

    def generate_random_callsign(self) -> str:
        """Generate a realistic random callsign."""
        prefix_len = random.randint(1, 2)
        prefix = "".join(random.choices(string.ascii_uppercase, k=prefix_len))
        digit = random.choice(string.digits)
        suffix_len = random.randint(1, 3)
        suffix = "".join(random.choices(string.ascii_uppercase, k=suffix_len))
        call = f"{prefix}{digit}{suffix}"
        if random.random() < 0.2: # Mobile operation
            call += f"/{random.choice(string.digits)}"
        return call

    def generate_phrase(self) -> str:
        """Generate a text from templates."""
        template = random.choice(config.PHRASE_TEMPLATES)
        call1 = self.generate_random_callsign()
        call2 = self.generate_random_callsign()
        # Generate random strings for name and city to avoid hallucination
        name = "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 6)))
        city = "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 8)))
        weather = random.choice(config.COMMON_WEATHER)
        temp = random.randint(-5, 35)
        rst = f"{random.randint(4, 5)}{random.randint(7, 9)}{random.randint(7, 9)}"
        # 599 -> 5NN conversion for realism
        rst = rst.replace('9', 'N')
        
        phrase = template.format(
            call=call1,
            call1=call1,
            call2=call2,
            name=name,
            city=city,
            rst=rst,
            weather=weather,
            temp=temp
        )
        # ワードの最後にも必ずスペースが入るようにし、挙動を一貫させる
        if not phrase.endswith(" "):
            phrase += " "
        return phrase

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        max_duration = 10.0
        is_phrase = random.random() < self.phrase_prob
        
        for attempt in range(5): # Retry if text is too long for WPM limits
            if is_phrase:
                text = self.generate_phrase()
                # Adaptive WPM for phrases to fit in 10s
                # Use a slightly smaller target to leave room for silence
                wpm = self.gen.estimate_wpm_for_target_frames(text, target_frames=int(max_duration * 0.9 * config.SAMPLE_RATE / config.HOP_LENGTH), min_wpm=self.min_wpm, max_wpm=self.max_wpm)
                
                # Verify if it fits
                timing = self.gen.generate_timing(text, wpm=wpm)
                if sum(t[1] for t in timing) < max_duration - 0.2:
                    break
                else:
                    # If it doesn't fit even at max_wpm, we'll retry with another phrase
                    if wpm >= self.max_wpm and attempt < 4:
                        continue
                    break
            else:
                # Decide WPM first to constrain length
                wpm = random.randint(self.min_wpm, self.max_wpm)

                # Generate random text
                max_allowed_len = self.gen.estimate_max_chars_for_wpm(wpm, target_frames=int(max_duration * 0.9 * config.SAMPLE_RATE / config.HOP_LENGTH))
                # Ensure length doesn't exceed VRAM-safe limit for this WPM
                length = random.randint(self.min_len, max(self.min_len, min(self.max_len, max_allowed_len)))
                
                # Randomly choose from available tokens (chars + prosigns)
                # Filter out spaces for random choice, we will add them manually
                valid_tokens = self.gen.text_to_morse_tokens(self.chars) if isinstance(self.chars, str) else self.chars
                valid_tokens = [t for t in valid_tokens if t != ' ']
                
                # Create a list of tokens
                if self.focus_chars and random.random() < self.focus_prob:
                    # Weighted sampling: Include at least one focus char, and higher prob for others
                    # Mix focus chars and valid tokens
                    # focus_chars をトークンに分解する
                    focus_tokens = self.gen.text_to_morse_tokens(self.focus_chars)
                    focus_valid = [t for t in focus_tokens if t != ' ' and t in valid_tokens]
                    
                    if focus_valid:
                        # Ensure at least 50% are focus chars, and try to include DIFFERENT focus chars
                        k_focus = max(1, length // 2)
                        k_other = length - k_focus
                        
                        # focus_valid から可能な限り多様に選ぶ (LとRの両方を入れるため、および Prosigns のため)
                        if len(focus_valid) > 1 and k_focus >= len(focus_valid):
                            tokens = random.sample(focus_valid, len(focus_valid)) # 必ず全種類1つは入れる
                            tokens += random.choices(focus_valid, k=k_focus - len(focus_valid))
                        else:
                            tokens = random.choices(focus_valid, k=k_focus)
                        
                        tokens += random.choices(valid_tokens, k=k_other)
                        random.shuffle(tokens)
                    else:
                        tokens = random.choices(valid_tokens, k=length)
                else:
                    tokens = random.choices(valid_tokens, k=length)
                
                # Join them, occasionally adding spaces
                text = ""
                for t in tokens:
                    text += t
                    # 単語間空白の学習機会を増やすため、挿入確率を 0.4 に引き上げ
                    if random.random() < 0.4:
                        text += " "
                
                # ワードの最後にも必ずスペースが入るようにし、挙動を一貫させる
                if not text.endswith(" "):
                    text += " "
                
                if not text.strip(): text = "CQ "
                
                # Verify if it fits
                timing = self.gen.generate_timing(text, wpm=wpm)
                if sum(t[1] for t in timing) < max_duration - 0.2:
                    break
                # else retry
        snr = random.uniform(self.min_snr_2500, self.max_snr_2500)
        
        # Determine jitter and weight based on curriculum settings
        jitter = random.uniform(0, self.jitter_max)
        # weight is centered at 1.0, variation is +/- weight_var
        weight = 1.0 + random.uniform(-self.weight_var, self.weight_var)
        
        fading_speed = random.uniform(self.fading_speed_min, self.fading_speed_max)
        
        frequency = random.uniform(self.min_freq, self.max_freq)

        # Randomly apply TX lowpass filter to soften edges
        tx_lowpass = None
        rise_time = 0.005 # Default
        if random.random() < self.tx_lowpass_prob:
            # Cutoff is typically somewhere above the carrier frequency.
            # 0.8x to 2.5x frequency covers from "muffled" to "standard".
            tx_lowpass = frequency * random.uniform(0.8, 2.5)
            # Also soften the rise time itself
            rise_time = random.uniform(0.005, self.rise_time_max)
        
        # New Augmentations based on dataset probabilities (set by curriculum)
        drift_hz = 0.0
        if random.random() < self.drift_prob:
            drift_hz = random.uniform(1.0, 15.0)
            
        qrn_strength = 0.0
        if random.random() < self.qrn_prob:
            # Strength relative to signal (mark_power=0.5)
            qrn_strength = random.uniform(0.5, 3.0)
            
        agc_enabled = random.random() < self.agc_prob
        
        multipath_delay = 0.0
        if random.random() < self.multipath_prob:
            multipath_delay = random.uniform(10.0, 50.0)
            
        clipping_threshold = 1.0
        if random.random() < self.clipping_prob:
            clipping_threshold = random.uniform(0.3, 0.8)

        waveform, label, signal_labels, boundary_labels = generate_sample(
            text, wpm=wpm, snr_2500=snr, jitter=jitter, weight=weight,
            fading_speed=fading_speed, min_fading=self.min_fading,
            frequency=frequency,
            tx_lowpass=tx_lowpass,
            rise_time=rise_time,
            min_gain_db=self.min_gain_db,
            drift_hz=drift_hz,
            qrn_strength=qrn_strength,
            qrm_prob=self.qrm_prob,
            impulse_prob=self.impulse_prob,
            agc_enabled=agc_enabled,
            multipath_delay=multipath_delay,
            clipping_threshold=clipping_threshold
        )
        # Return wpm as well so the trainer can use it for adaptive space reconstruction
        return waveform, label, wpm, signal_labels, boundary_labels, is_phrase

if __name__ == "__main__":
    sample_text = "CQ DE KILO CODE K"
    sample_rate = config.SAMPLE_RATE
    print(f"Generating sample: {sample_text}")
    
    waveform, label, signal_labels, _ = generate_sample(sample_text, wpm=25, snr_2500=20, sample_rate=sample_rate)
    
    output_file = "sample_cw.wav"
    # Convert back to numpy for scipy saving
    wf_np = waveform.numpy()
    scipy.io.wavfile.write(output_file, sample_rate, (wf_np * 32767).astype(np.int16))
    
    print(f"Saved to {output_file} using scipy")
    
    # Test Dataset
    dataset = CWDataset(num_samples=5)
    wf, lbl, wpm, sig = dataset[0]
    print(f"Dataset test - Waveform shape: {wf.shape}, Label: {lbl}, Signal shape: {sig.shape}")