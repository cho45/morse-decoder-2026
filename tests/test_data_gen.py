import torch
import numpy as np
import config
from data_gen import generate_sample, CWDataset, MorseGenerator

def test_generate_sample():
    # TEST -> - . ... - (T E S T)
    # reconstructed_text should match "TEST "
    waveform, label, signal_labels, boundary_labels = generate_sample("TEST", wpm=20, snr_2500=20)
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert isinstance(signal_labels, torch.Tensor)
    assert isinstance(boundary_labels, torch.Tensor)
    assert label == "TEST "

def test_cw_dataset():
    num_samples = 5
    dataset = CWDataset(num_samples=num_samples)
    assert len(dataset) == num_samples

    item = dataset[0]
    assert len(item) == 6
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = item
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert isinstance(wpm, int)
    assert len(label) > 0

def test_data_no_overflow():
    """Verify that the signal never overflows the max_duration (10s)."""
    gen = MorseGenerator(sample_rate=config.SAMPLE_RATE)
    text = "CQ CQ CQ DE KILO CODE " * 10
    wpm = 10
    max_duration = 10.0

    waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, max_duration=max_duration)
    
    # Check if the waveform length is exactly max_duration
    assert len(waveform) == int(max_duration * config.SAMPLE_RATE)
    
    # Check if the last frame is NOT a Dit or Dah signal
    # signal_labels mapping: 0: Background, 1: Dit, 2: Dah, 3: Inter-word
    assert signal_labels[-1] not in [1, 2], f"Signal overflowed at the end of the window: {signal_labels[-1]}"

if __name__ == '__main__':
    test_generate_sample()
    test_cw_dataset()
    test_data_no_overflow()