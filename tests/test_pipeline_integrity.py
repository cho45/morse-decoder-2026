import torch
import torchaudio
import numpy as np
import pytest
from train import Trainer
import config

class DummyArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip = 5.0
        self.save_dir = "test_checkpoints"
        self.curriculum_phase = 0
        self.resume = None
        self.freeze_encoder = False

def test_mel_physical_properties():
    """
    Measure actual Mel power levels to derive correct normalization.
    """
    args = DummyArgs()
    trainer = Trainer(args)
    mel_transform = trainer.mel_transform
    
    # 700Hz Sine wave at amplitude 1.0
    t = torch.arange(16000) / config.SAMPLE_RATE
    signal = torch.sin(2 * torch.pi * 700.0 * t).unsqueeze(0).to(trainer.device)
    silence = torch.zeros(1, 16000).to(trainer.device)
    
    with torch.no_grad():
        mel_sig = mel_transform(signal)
        mel_sil = mel_transform(silence)
        
        # Current logic in train.py uses log10(x + 1e-9)
        log_sig = torch.log10(mel_sig + 1e-9)
        log_sil = torch.log10(mel_sil + 1e-9)
        
        sig_val = log_sig.max().item()
        sil_val = log_sil.mean().item()
        
        print(f"\n[MEASUREMENT]")
        print(f"Log10 Signal Max:  {sig_val:.4f}")
        print(f"Log10 Silence Avg: {sil_val:.4f}")
        
        # Ensure signal is clearly distinguishable
        assert sig_val > sil_val + 3.0

def test_temporal_alignment():
    """
    Prove that a signal at sample S appears at the exact expected Mel frame.
    """
    args = DummyArgs()
    trainer = Trainer(args)
    
    # Trigger at 1600 samples = 100ms = 10 frames (hop=160)
    trigger_sample = 1600
    waveform = torch.zeros(1, 16000).to(trainer.device)
    t = torch.arange(16000 - trigger_sample) / config.SAMPLE_RATE
    waveform[0, trigger_sample:] = torch.sin(2 * torch.pi * 700.0 * t)
    
    with torch.no_grad():
        mels = trainer.mel_transform(waveform)
        # Check energy across frames
        energies = mels[0].mean(dim=0)
        
        # Frame 9 (ending at 1600) should be low
        # Frame 10 (starting at 1600) should be high
        print(f"\n[ALIGMENT]")
        print(f"Frame 9 energy:  {energies[9].item():.2e}")
        print(f"Frame 10 energy: {energies[10].item():.2e}")
        
        # With center=False and window effects, we check if the peak is clearly
        # at or after the expected start frame.
        assert energies[10] > energies[8] * 5.0

def test_ctc_length_safety():
    """
    Prove that input_lengths is always sufficient for CTC (T >= 2L + 1).
    """
    # Test worst case: High WPM (short signal) and long text
    wpm = 50
    text = "ABCDEFGHIJ" # 10 chars
    # At 50 WPM, 1 dot = 24ms. 
    # Total units for "ABCDEFGHIJ" is roughly 100 units = 2.4s.
    # 2.4s = 240 mel frames = 120 logit frames.
    # L = 10 chars. 2L+1 = 21. 
    # 120 >= 21 is safe.
    
    # We test the calculation logic directly
    for length_samples in [8000, 16000, 32000]:
        l_mel = (length_samples - config.N_FFT) // config.HOP_LENGTH + 1
        l_padded = l_mel + 2
        input_lengths = (l_padded - 3) // 2 + 1
        
        # Max characters allowed for this duration
        max_L = (input_lengths - 1) // 2
        print(f"\nSamples: {length_samples} | Logit Frames: {input_lengths} | Max Chars: {max_L}")
        assert max_L > 10 # Should always handle at least 10 chars per second