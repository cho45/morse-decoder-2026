import torch
import numpy as np
import pytest
from data_gen import generate_sample, CWDataset
import config

def test_generate_sample_with_signal_labels():
    text = "K" # -.-
    waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=20, snr_2500=100)
    
    assert isinstance(waveform, torch.Tensor)
    assert label == text + " "
    assert isinstance(signal_labels, torch.Tensor)
    assert isinstance(boundary_labels, torch.Tensor)
    
    # Calculate expected number of mel frames
    # Mel frames: floor((L - N_FFT) / HOP_LENGTH) + 1
    num_samples = waveform.shape[0]
    expected_frames = (num_samples - config.N_FFT) // config.HOP_LENGTH + 1
    
    assert signal_labels.shape[0] == expected_frames
    assert signal_labels.dtype == torch.float32
    assert torch.all((signal_labels >= 0) & (signal_labels < config.NUM_SIGNAL_CLASSES))

def test_dataset_returns_signal_labels():
    dataset = CWDataset(num_samples=2)
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[0]
    
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert isinstance(wpm, int)
    assert isinstance(signal_labels, torch.Tensor)
    
    num_samples = waveform.shape[0]
    expected_frames = (num_samples - config.N_FFT) // config.HOP_LENGTH + 1
    assert signal_labels.shape[0] == expected_frames

def test_signal_labels_alignment():
    # Very simple case: long dash
    text = "T" # -
    waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=10, snr_2500=100)
    
    # Check if we have 2s (Dah) in the signal_labels for 'T'
    assert torch.any(signal_labels == 2)
    
    # The beginning and end (pre/post silence) should be 0
    # pre_silence is at least 100ms (10 frames)
    assert torch.all(signal_labels[:5] == 0)
    assert torch.all(signal_labels[-5:] == 0)