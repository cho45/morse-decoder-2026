import torch
import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import Trainer
import config
from unittest.mock import patch

class MockArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.epochs = 1
        self.batch_size = 2
        self.lr = 1e-4
        self.save_dir = "checkpoints"
        self.resume = None
        self.curriculum_phase = 0
        self.freeze_encoder = False
        self.reset_ctc_head = False
        self.d_model = config.D_MODEL
        self.n_head = config.N_HEAD
        self.num_layers = config.NUM_LAYERS
        self.dropout = 0.1

class TestCollateLookahead(unittest.TestCase):
    def setUp(self):
        self.args = MockArgs()
        # Mock cuda to force cpu
        with patch('torch.cuda.is_available', return_value=False):
            self.trainer = Trainer(self.args)
        # Ensure everything is on CPU
        self.trainer.device = torch.device("cpu")
        self.trainer.spec_transform.to("cpu")
        self.trainer.model.to("cpu")
    
    def tearDown(self):
        pass

    def test_collate_fn_lookahead_shift(self):
        """
        Test if collate_fn correctly shifts labels to account for lookahead padding.
        
        Expected behavior:
        Input Waveform:  [Signal (T)] + [Pad (Lookahead)]
        Target Label:    [Pad (Lookahead/Subsampled)] + [Signal Label]
        
        This ensures that when the model sees Input[t], it is predicting Label[t - Lookahead].
        Or equivalently, Input[t + Lookahead] is used to predict Label[t].
        """
        
        # Create a dummy batch
        # 1. Waveform: Simple sine wave, 1 second
        sample_rate = config.SAMPLE_RATE
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * 3.14159 * 700 * t)
        
        # 2. Labels
        text = "E"
        wpm = 20
        is_phrase = False
        
        # Calculate expected frames
        # Mel frames = (L - N_FFT) // HOP + 1
        l_total = len(waveform)
        l_mel = (l_total - config.N_FFT) // config.HOP_LENGTH + 1
        
        # Signal Labels (Mel frame resolution)
        # Let's say it's all '1' (Dit) for simplicity, though physically impossible
        signal_labels = torch.ones(l_mel, dtype=torch.long)
        boundary_labels = torch.zeros(l_mel, dtype=torch.float32)
        
        batch = [(waveform, text, wpm, signal_labels, boundary_labels, is_phrase)]
        
        # Run collate_fn
        padded_waveforms, _, lengths, _, _, _, padded_signals, padded_boundaries, _ = self.trainer.collate_fn(batch)
        
        # --- Verification ---
        
        # 1. Check Input Padding (Lookahead)
        lookahead_samples = config.LOOKAHEAD_FRAMES * config.HOP_LENGTH
        expected_input_len = len(waveform) + lookahead_samples
        self.assertEqual(padded_waveforms.shape[1], expected_input_len, "Input waveform length mismatch")
        
        # Verify padding is zeros
        padding_area = padded_waveforms[0, len(waveform):]
        self.assertTrue(torch.all(padding_area == 0), "Input padding area should be zeros")

        # 2. Check Label Shifting (The Core Issue)
        # Calculate how many frames correspond to lookahead in the label domain (after subsampling)
        subsampling_rate = config.SUBSAMPLING_RATE
        lookahead_frames_mel = config.LOOKAHEAD_FRAMES
        lookahead_frames_sub = lookahead_frames_mel // subsampling_rate
        
        print(f"\nDebug Info:")
        print(f"Lookahead frames (Mel): {lookahead_frames_mel}")
        print(f"Subsampling rate: {subsampling_rate}")
        print(f"Expected Label Shift (Frames): {lookahead_frames_sub}")
        
        # Check if the beginning of the label is padded (shifted)
        # The first 'lookahead_frames_sub' frames should be 0 (Background/Space),
        # even though our input signal_labels was all 1s.
        
        first_frames = padded_signals[0, :lookahead_frames_sub]
        print(f"First {lookahead_frames_sub} frames of label: {first_frames}")
        
        # Check if they are all 0 (Space)
        is_shifted = torch.all(first_frames == 0)
        
        if not is_shifted:
            print("FAIL: Labels are NOT shifted!")
        else:
            print("PASS: Labels are correctly shifted.")

        self.assertTrue(is_shifted, f"Labels should be right-shifted by {lookahead_frames_sub} frames. Found: {first_frames}")
        
        # Also check boundary labels
        first_bound = padded_boundaries[0, :lookahead_frames_sub]
        self.assertTrue(torch.all(first_bound == 0), "Boundary labels should also be right-shifted.")

if __name__ == '__main__':
    unittest.main()