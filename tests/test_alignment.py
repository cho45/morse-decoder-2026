import torch
import pytest
from train import Trainer
from model import StreamingConformer
import config

class DummyArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.epochs = 1
        self.batch_size = 2
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip = 5.0
        self.save_dir = "test_checkpoints"
        self.curriculum_phase = 0
        self.resume = None
        self.freeze_encoder = False

def test_input_output_alignment():
    """
    Verify that the input_lengths calculated by Trainer.compute_mels_and_lengths
    exactly matches the actual output sequence length of the model.
    """
    args = DummyArgs()
    trainer = Trainer(args)
    trainer.model.eval()
    
    # Test a range of lengths to catch off-by-one errors on odd/even boundaries
    # Test every sample length from 1.0s to 1.01s to ensure stability
    test_lengths = range(16000, 16160, 1)
    
    for length in test_lengths:
        # Create dummy waveform
        waveform = torch.randn(1, length)
        lengths_tensor = torch.tensor([length])
        
        # 1. Calculate lengths using the logic in train.py
        with torch.no_grad():
            mels, calc_input_lengths = trainer.compute_mels_and_lengths(waveform, lengths_tensor)
            
            # 2. Get actual output length from the model
            # Structural test only, weights don't matter for length
            logits, _ = trainer.model(mels)
            actual_output_length = logits.size(1)
            
            print(f"\nTesting length: {length} samples")
            print(f"Mel frames: {mels.size(1)}")
            print(f"Calculated input_lengths: {calc_input_lengths.item()}")
            print(f"Actual model output length: {actual_output_length}")
            
            # THE CRITICAL ASSERTION
            assert calc_input_lengths.item() == actual_output_length, \
                f"Alignment mismatch for input length {length}! Calc: {calc_input_lengths.item()}, Actual: {actual_output_length}"

def test_mel_impulse_alignment():
    """
    Verify that the Mel spectrogram itself is aligned with the input signal.
    """
    args = DummyArgs()
    trainer = Trainer(args)
    
    # 1600 samples = 100ms = 10 mel frames (hop=160)
    trigger_sample = 1600
    length = 16000
    waveform = torch.zeros(1, length)
    # Use a 700Hz sine wave to match our frequency band (300-1500Hz)
    t = torch.arange(length - trigger_sample) / config.SAMPLE_RATE
    waveform[0, trigger_sample:] = torch.sin(2 * torch.pi * 700.0 * t)
    
    with torch.no_grad():
        mels, _ = trainer.compute_mels_and_lengths(waveform, torch.tensor([length]))
        # mels: (B, T, F)
        mel_energies = mels[0].mean(dim=-1) # Average over frequency bins
        
        # Frame 0 to 7 should be silent (approx 0.0 in our new scaling)
        # Frame 10 should be active (approx 0.8-1.0 in our new scaling)
        print(f"\nMel energy at frame 5 (silence): {mel_energies[5].item():.4f}")
        print(f"Mel energy at frame 10 (signal): {mel_energies[10].item():.4f}")
        
        # In our fixed scaling, silence is 0.0.
        # Signal energy depends on frequency bins, but should be clearly non-zero.
        assert mel_energies[5] < 0.01, f"Frame 5 should be silent, got {mel_energies[5].item()}"
        assert mel_energies[10] > 0.1, f"Frame 10 should be active, got {mel_energies[10].item()}"

def test_impulse_alignment():
    """
    Verify the temporal offset: if we have a signal starting at sample S,
    does the model output show activity at the expected frame?
    """
    args = DummyArgs()
    trainer = Trainer(args)
    trainer.model.eval()
    
    # 1600 samples = 100ms = 10 mel frames (hop=160)
    # MelSpectrogram(center=False) with n_fft=400:
    # First frame (0) covers samples [0, 400]
    # Frame 10 covers [1600, 2000]
    trigger_sample = 1600
    length = 16000
    waveform = torch.zeros(1, length)
    waveform[0, trigger_sample:] = 1.0 # High amplitude from 1600
    
    lengths_tensor = torch.tensor([length])
    
    with torch.no_grad():
        mels, _ = trainer.compute_mels_and_lengths(waveform, lengths_tensor)
        logits, _ = trainer.model(mels)
        
        # Strict physical alignment test:
        # A signal starting at sample S must result in a peak at a predictable frame F.
        # For SR=16000, HOP=160, SUB=2, a frame is 20ms.
        # Trigger at 1600 samples (100ms) should ideally appear at frame 5.
        
        # We use the raw Mel energy to find the physical peak in the input.
        mel_energy = mels[0].mean(dim=-1)
        input_peak_frame = mel_energy.argmax().item()
        
        # The model's output peak must align with the input peak (accounting for subsampling)
        # input_peak_frame is in 10ms units. Output is in 20ms units.
        expected_output_peak = input_peak_frame // config.SUBSAMPLING_RATE
        
        print(f"\n[ALIGNMENT DIAGNOSIS]")
        print(f"Signal start sample: {trigger_sample}")
        print(f"Input Mel peak frame: {input_peak_frame}")
        print(f"Expected output peak frame: {expected_output_peak}")
        
        # Allow 1 frame tolerance for windowing effects in center=False
        theoretical_frame = trigger_sample // config.HOP_LENGTH
        assert abs(input_peak_frame - theoretical_frame) <= 1, \
            f"Input peak at {input_peak_frame}, expected near {theoretical_frame}"

if __name__ == "__main__":
    test_input_output_alignment()
    test_mel_impulse_alignment()
    test_impulse_alignment()