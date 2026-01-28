import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import CWDataset
import config

def test_low_wpm_limit():
    # Test for very low WPM
    dataset = CWDataset(num_samples=100, min_wpm=10, max_wpm=10, min_len=5, max_len=20)
    
    print(f"Testing with WPM=10, TARGET_FRAMES={config.TARGET_FRAMES}")
    
    total_samples = 0
    max_frames = 0
    
    for i in range(20):
        waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[i]
        num_frames = (len(waveform) - config.N_FFT) // config.HOP_LENGTH + 1
        print(f"Sample {i}: WPM={wpm}, Length={len(label)}, Frames={num_frames}, Text='{label}'")
        
        max_frames = max(max_frames, num_frames)
        total_samples += 1
        
        # Check if frames are within reasonable limit (allowing some buffer for silence/padding)
        # TARGET_FRAMES=1000 is about 10 seconds.
        # MorseGenerator.generate_waveform adds pre_silence(0.1-0.5) and post_silence(0.55).
        # So it should be around 1000 + 100 frames.
        assert num_frames < config.TARGET_FRAMES + 200, f"Frame length {num_frames} exceeds limit for WPM {wpm}"
        
        # At 10 WPM, 1 unit = 1.2/10 = 0.12s.
        # PARIS standard (50 units) = 6s.
        # 10 seconds allows about 1.6 words = 8-10 chars.
        assert len(label) <= 15, f"Text length {len(label)} too long for 10 WPM"

    print(f"Max frames observed: {max_frames}")
    print("Low WPM limit test passed!")

def test_high_wpm_limit():
    # Test for high WPM
    dataset = CWDataset(num_samples=100, min_wpm=40, max_wpm=40, min_len=5, max_len=20)
    
    print(f"\nTesting with WPM=40, TARGET_FRAMES={config.TARGET_FRAMES}")
    
    for i in range(5):
        waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[i]
        num_frames = (len(waveform) - config.N_FFT) // config.HOP_LENGTH + 1
        print(f"Sample {i}: WPM={wpm}, Length={len(label)}, Frames={num_frames}, Text='{label}'")
        
        # At 40 WPM, we should be able to fit more characters
        # (Actually max_len is 20, so it will be capped by max_len)
        assert num_frames < config.TARGET_FRAMES + 200

    print("High WPM limit test passed!")

if __name__ == "__main__":
    test_low_wpm_limit()
    test_high_wpm_limit()