import torch
import numpy as np
import config
from config import (
    SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE
)
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
    """Verify that the signal never overflows the max_duration."""
    gen = MorseGenerator(sample_rate=config.SAMPLE_RATE)
    text = "CQ CQ CQ DE KILO CODE " * 10
    wpm = 10
    max_duration = config.TRAIN_DURATION

    waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, max_duration=max_duration)
    
    # Check if the waveform length is exactly max_duration
    expected_samples = int(max_duration * config.SAMPLE_RATE)
    assert len(waveform) == expected_samples, f"Waveform length {len(waveform)} mismatch with expected {expected_samples}"
    
    # Check if the last frame is NOT a Dit or Dah signal
    # signal_labels mapping: SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE
    assert signal_labels[-1] not in [SIG_ID_DIT, SIG_ID_DAH], f"Signal overflowed at the end of the window: {signal_labels[-1]}"

if __name__ == '__main__':
    test_generate_sample()
    test_cw_dataset()
    test_data_no_overflow()