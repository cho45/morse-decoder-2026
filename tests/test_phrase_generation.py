import sys
import os
sys.path.append(os.getcwd())

import torch
from data_gen import CWDataset
import config

def test_phrase_generation():
    print("Testing phrase generation and adaptive WPM...")
    # Enable phrase generation
    dataset = CWDataset(num_samples=20, phrase_prob=1.0)
    
    for i in range(10):
        waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[i]
        num_frames = (waveform.shape[0] - config.N_FFT) // config.HOP_LENGTH + 1
        print(f"Sample {i+1}:")
        print(f"  Text: {label}")
        print(f"  WPM:  {wpm}")
        print(f"  Frames: {num_frames}")
        
        # Check if adaptive WPM is working
        # (With shorter phrases, WPM might be lower than 25 but should still fit in frames)
        if "/" in label:
            print("  [OK] Mobile operation callsign detected.")

    print("\nTesting random text with normal WPM...")
    dataset_random = CWDataset(num_samples=10, phrase_prob=0.0, min_wpm=20, max_wpm=20)
    for i in range(5):
        waveform, label, wpm, _, _, is_phrase = dataset_random[i]
        print(f"  Text: {label}, WPM: {wpm}")
        assert wpm == 20

if __name__ == "__main__":
    test_phrase_generation()