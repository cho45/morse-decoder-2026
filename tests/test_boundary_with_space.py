import unittest
import torch
import numpy as np
from data_gen import MorseGenerator, generate_sample
import config

class TestBoundaryWithSpace(unittest.TestCase):
    def test_boundary_count_with_space(self):
        # "A B " は 'A', ' ', 'B', ' ' の4文字として扱われるべき
        # したがって、境界ラベルの塊（5フレーム連続）は4つ存在するはず
        wpm = 20
        text = "A B "
        waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_2500=100)
        
        bound = boundary_labels.numpy()
        
        # 境界ラベルが 0 -> 1 に変化する箇所をカウント
        diff = np.diff(bound, prepend=0)
        boundary_indices = np.where(diff > 0)[0]
        num_boundaries = len(boundary_indices)
        
        print(f"DEBUG: Text='{text}', Num boundaries found: {num_boundaries}")
        for i, idx in enumerate(boundary_indices):
            print(f"  Boundary {i+1}: frame {idx}, signal_class around: {signal_labels.numpy()[max(0, idx-5):idx+5]}")

        self.assertEqual(num_boundaries, 4, f"Expected 4 boundaries for '{text}', but found {num_boundaries}")

if __name__ == '__main__':
    unittest.main()