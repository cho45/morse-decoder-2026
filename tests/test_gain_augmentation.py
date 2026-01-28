import pytest
import torch
import numpy as np
import random
from data_gen import generate_sample, CWDataset

def test_generate_sample_gain_db():
    # min_gain_db=0.0 (default) should always have max amplitude 1.0
    waveform, _, _, _ = generate_sample("TEST", min_gain_db=0.0, snr_db=100)
    assert np.max(np.abs(waveform.numpy())) == pytest.approx(1.0)

    # min_gain_db=-40.0 should have varying max amplitudes in log scale
    max_amps = []
    for _ in range(50):
        waveform, _, _, _ = generate_sample("TEST", min_gain_db=-40.0, snr_db=100)
        max_amps.append(np.max(np.abs(waveform.numpy())))
    
    max_amps = np.array(max_amps)
    max_amps_db = 20 * np.log10(max_amps + 1e-12)
    
    print(f"\nMax amplitudes dB (min_gain_db=-40.0): {max_amps_db.mean():.2f}dB avg, {max_amps_db.min():.2f}dB min")
    
    # Check if we have some very small values (around -40dB = 0.01)
    assert np.any(max_amps < 0.05) 
    assert np.any(max_amps > 0.5)
    # Distribution should be somewhat uniform in dB
    assert np.std(max_amps_db) > 5.0

def test_cw_dataset_gain_db():
    # Test if CWDataset passes min_gain_db to generate_sample
    dataset = CWDataset(num_samples=20, min_gain_db=-20.0, min_snr=100, max_snr=100)
    max_amps = []
    for i in range(len(dataset)):
        waveform, _, _, _, _, _ = dataset[i]
        max_amps.append(np.max(np.abs(waveform.numpy())))
    
    max_amps = np.array(max_amps)
    max_amps_db = 20 * np.log10(max_amps + 1e-12)
    
    print(f"\nDataset Max amplitudes dB (min_gain_db=-20.0): {max_amps_db}")
    
    assert np.all(max_amps_db >= -20.5) # Allow for small float errors
    assert np.all(max_amps_db <= 0.5)
    assert np.std(max_amps_db) > 3.0

if __name__ == "__main__":
    test_generate_sample_gain_db()
    test_cw_dataset_gain_db()
    print("Gain augmentation (dB scale) tests passed!")