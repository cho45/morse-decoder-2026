import torch
import numpy as np
import data_gen
import config
import unittest

class TestExtremeCurriculum(unittest.TestCase):
    def test_negative_snr_generation(self):
        text = "CQ"
        snr = -15.0
        fading_speed = 0.5
        
        waveform, label, signal_labels, boundary_labels = data_gen.generate_sample(
            text, wpm=20, snr_db=snr, fading_speed=fading_speed
        )
        
        self.assertEqual(label, text)
        self.assertTrue(torch.is_tensor(waveform))
        # Check if waveform is normalized
        self.assertLessEqual(torch.max(torch.abs(waveform)), 1.0)
        print(f"Generated -15dB sample, waveform shape: {waveform.shape}")

    def test_dataset_extreme_params(self):
        dataset = data_gen.CWDataset(
            num_samples=5,
            min_snr=-15.0, max_snr=-10.0,
            fading_speed_min=0.1, fading_speed_max=0.5
        )
        
        waveform, label, wpm, signal_labels, boundary_labels = dataset[0]
        self.assertTrue(torch.is_tensor(waveform))
        print(f"Dataset extreme sample generated: SNR range [-15, -10], label: {label}")

if __name__ == "__main__":
    unittest.main()