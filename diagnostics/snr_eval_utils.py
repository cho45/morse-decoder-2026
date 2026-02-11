import torch
import numpy as np
import random
import string
import os
import sys
from typing import List, Tuple
from torch.utils.data import Dataset
from collections import defaultdict

# Add parent directory to path to import modules if not already added
# (Assumes this file is in diagnostics/ subdirectory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_gen import generate_sample, CWDataset, MorseGenerator

class SyntheticMorseDataset(Dataset):
    def __init__(self, samples_per_snr: int, snrs: List[float], dataset: CWDataset, wpm: int = 15,
                 random_freq: bool = False, type: str = 'random',
                 fading_speed: float = 0.0, min_fading: float = 1.0,
                 qrm_prob: float = 0.0, impulse_prob: float = 0.0):
        self.samples_per_snr = samples_per_snr
        self.snrs = snrs
        self.total_samples = samples_per_snr * len(snrs)
        self.dataset = dataset
        self.wpm = wpm
        self.random_freq = random_freq
        self.type = type
        self.fading_speed = fading_speed
        self.min_fading = min_fading
        self.qrm_prob = qrm_prob
        self.impulse_prob = impulse_prob
        self.gen = MorseGenerator()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which SNR this sample belongs to
        snr_idx = idx // self.samples_per_snr
        current_snr = self.snrs[snr_idx]
        
        if self.type == 'random':
            text = self._generate_random_text()
        else:
            text = self._generate_packed_phrase()
        
        freq = random.uniform(config.MIN_FREQ, config.MAX_FREQ) if self.random_freq else 700.0
        
        # generate_sample handles truncation/padding
        waveform, actual_text, _, _ = generate_sample(
            text=text, wpm=self.wpm, snr_2500=current_snr, frequency=freq,
            jitter=0.0, weight=1.0, fading_speed=self.fading_speed, min_fading=self.min_fading,
            qrm_prob=self.qrm_prob, impulse_prob=self.impulse_prob
        )
        
        return waveform, actual_text, self.wpm, freq, current_snr

    def _generate_random_text(self) -> str:
        """Generate random text that fits in 10s at given WPM with high density."""
        # Target about 80% of 10s
        max_chars = self.gen.estimate_max_chars_for_wpm(self.wpm, target_frames=800)
        chars = string.ascii_uppercase + string.digits
        text = "".join(random.choices(chars, k=max_chars))
        # Add some spaces
        text_with_spaces = ""
        for c in text:
            text_with_spaces += c
            if random.random() < 0.2:
                text_with_spaces += " "
        return text_with_spaces.strip() + " "

    def _generate_packed_phrase(self) -> str:
        """Generate multiple phrases concatenated to fit in 10s."""
        max_duration = 10.0
        text = self.dataset.generate_phrase()
        
        for _ in range(3):
            next_phrase = self.dataset.generate_phrase()
            if self.gen.estimate_duration(text + " " + next_phrase, self.wpm) < max_duration - 1.0:
                text += " " + next_phrase
            else:
                break
        return text.strip() + " "
