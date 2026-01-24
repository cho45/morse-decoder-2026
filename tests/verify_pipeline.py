import torch
import numpy as np
from train import Trainer
import config

def verify_normalization_and_alignment():
    """
    Proves the mathematical correctness of the preprocessing pipeline.
    """
    class Args:
        samples_per_epoch = 10
        lr = 1e-4
        weight_decay = 1e-4
        grad_clip = 5.0
        freeze_encoder = False
    
    trainer = Trainer(Args())
    
    # 1. Generate pure 700Hz tone
    duration_samples = 16000
    t = torch.arange(duration_samples) / config.SAMPLE_RATE
    # Pulse from 500ms to 600ms (Frame 50 to 60)
    waveform = torch.zeros(1, duration_samples)
    waveform[0, 8000:9600] = torch.sin(2 * torch.pi * 700.0 * t[8000:9600])
    
    with torch.no_grad():
        # Apply the logic to be implemented in train.py
        mels = trainer.mel_transform(waveform.to(trainer.device))
        mels = torch.log10(mels + 1e-9)
        
        # PROOF: Scaling must map -9.0 to -1.0 and -3.0 to 1.0
        mels = (mels + 6.0) / 3.0
        mels = torch.clamp(mels, -1.0, 1.0)
        
        # Check frame 20 (silence) and frame 55 (middle of pulse)
        # Frame 50 is at 500ms (8000 samples)
        val_silence = mels[0, :, 20].mean().item()
        val_signal = mels[0, :, 55].mean().item()
        
        print(f"\n[VERIFICATION]")
        print(f"Silence Value (expected -1.0): {val_silence:.4f}")
        print(f"Signal Value  (expected  1.0): {val_signal:.4f}")
        
        assert abs(val_silence - (-1.0)) < 0.01
        assert abs(val_signal - 1.0) < 0.1

if __name__ == "__main__":
    verify_normalization_and_alignment()
    print("Pipeline math verified successfully!")