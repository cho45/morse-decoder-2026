import torch
import numpy as np
import data_gen
import config
import pytest

def test_qrn_generation():
    sim = data_gen.HFChannelSimulator()
    waveform = np.zeros(16000) # 1 sec
    qrn_waveform = sim.apply_qrn(waveform, strength=1.0)
    assert np.max(np.abs(qrn_waveform)) > 0
    assert len(qrn_waveform) == 16000

def test_agc_effect():
    sim = data_gen.HFChannelSimulator()
    # Create a signal that suddenly gets very loud
    t = np.arange(16000) / 16000
    signal = np.sin(2 * np.pi * 700 * t)
    signal[8000:] *= 10.0 # 10x stronger after 0.5s
    
    agc_waveform = sim.apply_agc(signal, attack_ms=5, release_ms=100)
    
    # After AGC, the latter part should be suppressed
    # The peak of the second half should be much less than 10.0
    assert np.max(np.abs(agc_waveform[8100:])) < 5.0
    # And it should be greater than 0
    assert np.max(np.abs(agc_waveform[8100:])) > 0

def test_drift_generation():
    gen = data_gen.MorseGenerator()
    timing = [(1, 0.1), (3, 0.1), (2, 0.3)] # Dit, space, Dah
    # With drift
    waveform_drift = gen.generate_waveform(timing, frequency=700, drift_hz=50)
    assert len(waveform_drift) > 0

def test_dataset_integration():
    dataset = data_gen.CWDataset(num_samples=10)
    # Enable all augmentations by overriding config temporarily if needed
    # but the defaults in config.py already have some probabilities.
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[0]
    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 1

if __name__ == "__main__":
    test_qrn_generation()
    test_agc_effect()
    test_drift_generation()
    test_dataset_integration()
    print("All new augmentation tests passed!")