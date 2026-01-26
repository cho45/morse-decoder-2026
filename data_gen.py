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
            if not word_raw: continue
            tokens = self.text_to_morse_tokens(word_raw)
            
            for j, char in enumerate(tokens):
                code = MORSE_DICT.get(char, "")
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
                                      min_wpm: int = 15, max_wpm: int = 45) -> int:
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

    def generate_waveform(self, timing: List[Tuple[int, float]], frequency: float = 700.0,
                          waveform_type: str = 'sine', rise_time: float = 0.005, wpm: int = 20) -> np.ndarray:
        """
        Convert timing sequence to audio waveform.
        """
        pre_silence = random.uniform(0.1, 0.5)
        post_silence = 0.55
        total_duration = sum(t[1] for t in timing) + pre_silence + post_silence
        total_samples = int(total_duration * self.sample_rate)
        waveform = np.zeros(total_samples)
        
        current_sample = int(pre_silence * self.sample_rate)
        for class_id, duration in timing:
            num_samples = int(duration * self.sample_rate)
            is_on = class_id in [1, 2]
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
            if class_id in [4, 5]: # Inter-char space or Inter-word space (文字の終了)
                # 空白の終了時点（＝次の要素の開始直前）を特定
                trigger_sample = time_ptr + duration_samples
                
                # 信号が完全に終了した後の最初のフレームを特定する。
                # torchaudio center=False では、フレーム m はサンプル m * HOP_LENGTH から始まる。
                # よって、trigger_sample // HOP_LENGTH が、そのサンプルを含むかそれ以降の最初のフレーム。
                trigger_frame = trigger_sample // config.HOP_LENGTH
                
                # サブサンプリング(2x)で消えないよう、2フレーム分立てる
                for offset in range(2):
                    if 0 <= trigger_frame + offset < num_frames:
                        boundary_frames[trigger_frame + offset] = 1.0
            
            time_ptr += duration_samples

        # シーケンスの最後（最後の文字の終了後、文字間空白分(3ユニット)待ってから確定）
        # 文字間空白を待つことで、信号との重なりを確実に避ける。
        char_space_samples = int(3 * dot_len_sec * self.sample_rate)
        trigger_sample_final = time_ptr + char_space_samples
        trigger_frame_final = trigger_sample_final // config.HOP_LENGTH
        for offset in range(2):
            if 0 <= trigger_frame_final + offset < num_frames:
                boundary_frames[trigger_frame_final + offset] = 1.0

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

    def apply_noise(self, waveform: np.ndarray, snr_db: float = 10.0, impulse_prob: float = 0.001) -> np.ndarray:
        """Apply AWGN and impulse noise."""
        # AWGN
        # SNR is defined based on the average power of the signal during the MARK (ON) state.
        # For a sine wave with amplitude 1.0, the power is 0.5.
        mark_power = 0.5
        noise_power = mark_power / (10**(snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
        
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
                    jitter: float = 0.0, weight: float = 1.0,
                    fading_speed: float = 0.1, min_fading: float = 0.05,
                    frequency: float = 700.0) -> Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
    gen = MorseGenerator(sample_rate=sample_rate)
    sim = HFChannelSimulator(sample_rate=sample_rate)
    
    # Human artifacts are now controlled by arguments
    
    timing = gen.generate_timing(text, wpm=wpm, jitter=jitter, weight=weight)
    waveform, signal_labels, boundary_labels = gen.generate_waveform(timing, frequency=frequency, wpm=wpm)
    
    # Only apply channel effects if SNR is below a certain threshold (e.g., 45dB)
    if snr_db < 45:
        waveform = sim.apply_fading(waveform, speed_hz=fading_speed, min_fading=min_fading)
        if random.random() > 0.5:
            waveform = sim.apply_qrm(waveform, snr_db=snr_db + random.uniform(0, 10))
        waveform = sim.apply_noise(waveform, snr_db=snr_db)
        # Apply filter centered at the signal frequency
        waveform = sim.apply_filter(waveform, center_freq=frequency)
    else:
        # Truly clean: no noise, no filter, no fading
        pass
    
    # Normalize
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
        
    return torch.from_numpy(waveform).float(), text, torch.from_numpy(signal_labels).float(), torch.from_numpy(boundary_labels).float()

class CWDataset(Dataset):
    def __init__(self, num_samples: int = 1000, min_wpm: int = 10, max_wpm: int = 40,
                 min_snr: float = 5.0, max_snr: float = 25.0,
                 jitter_max: float = 0.1, weight_var: float = 0.2,
                 allowed_chars: str = None, min_len: int = 5, max_len: int = 10,
                 focus_chars: str = None, focus_prob: float = 0.5,
                 fading_speed_min: float = 0.1, fading_speed_max: float = 0.1,
                 min_fading: float = 0.1,
                 phrase_prob: float = 0.0,
                 min_freq: float = 650.0,
                 max_freq: float = 750.0):
        self.num_samples = num_samples
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm
        self.min_snr = min_snr
        self.max_snr = max_snr
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
        
        return template.format(
            call=call1,
            call1=call1,
            call2=call2,
            name=name,
            city=city,
            rst=rst,
            weather=weather,
            temp=temp
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        is_phrase = random.random() < self.phrase_prob
        if is_phrase:
            text = self.generate_phrase()
            # Adaptive WPM for phrases to fit in TARGET_FRAMES
            wpm = self.gen.estimate_wpm_for_target_frames(text, target_frames=config.TARGET_FRAMES)
        else:
            # Generate random text
            length = random.randint(self.min_len, self.max_len)
            
            # Randomly choose from available tokens (chars + prosigns)
            # Filter out spaces for random choice, we will add them manually
            valid_tokens = [c for c in self.chars if c != ' ']
            
            # Create a list of tokens
            if self.focus_chars and random.random() < self.focus_prob:
                # Weighted sampling: Include at least one focus char, and higher prob for others
                # Mix focus chars and valid tokens
                focus_valid = [c for c in self.focus_chars if c != ' ' and c in valid_tokens]
                if focus_valid:
                    # Ensure at least 50% are focus chars, and try to include DIFFERENT focus chars
                    k_focus = max(1, length // 2)
                    k_other = length - k_focus
                    
                    # focus_valid から可能な限り多様に選ぶ (LとRの両方を入れるため)
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
        
        fading_speed = random.uniform(self.fading_speed_min, self.fading_speed_max)
        
        frequency = random.uniform(self.min_freq, self.max_freq)
        
        waveform, label, signal_labels, boundary_labels = generate_sample(
            text, wpm=wpm, snr_db=snr, jitter=jitter, weight=weight,
            fading_speed=fading_speed, min_fading=self.min_fading,
            frequency=frequency
        )
        # Return wpm as well so the trainer can use it for adaptive space reconstruction
        return waveform, label, wpm, signal_labels, boundary_labels, is_phrase

if __name__ == "__main__":
    sample_text = "CQ DE KILO CODE K"
    sample_rate = config.SAMPLE_RATE
    print(f"Generating sample: {sample_text}")
    
    waveform, label, signal_labels = generate_sample(sample_text, wpm=25, snr_db=15, sample_rate=sample_rate)
    
    output_file = "sample_cw.wav"
    # Convert back to numpy for scipy saving
    wf_np = waveform.numpy()
    scipy.io.wavfile.write(output_file, sample_rate, (wf_np * 32767).astype(np.int16))
    
    print(f"Saved to {output_file} using scipy")
    
    # Test Dataset
    dataset = CWDataset(num_samples=5)
    wf, lbl, wpm, sig = dataset[0]
    print(f"Dataset test - Waveform shape: {wf.shape}, Label: {lbl}, Signal shape: {sig.shape}")