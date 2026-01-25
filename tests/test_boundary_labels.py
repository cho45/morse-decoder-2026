import unittest
import torch
import numpy as np
from data_gen import MorseGenerator, generate_sample
import config

class TestBoundaryLabels(unittest.TestCase):
    def test_boundary_timing(self):
        # WPM=20 の時、1ユニット(dot_len)は 1.2 / 20 = 0.06s (60ms)
        # 文字間空白は 3ユニット = 180ms
        # 1フレーム(HOP_LENGTH)は 10ms
        # 文字終了(信号のONが終了)から 18フレーム後付近にラベルが立つはず
        wpm = 20
        text = "K" # -.-
        waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_db=100)
        
        # signal_labels: 0:BG, 1:Dit, 2:Dah, 3:Intra, 4:Inter, 5:Word
        sig = signal_labels.numpy()
        bound = boundary_labels.numpy()
        
        # 文字の最後のDah(2)を探す
        last_dah_idx = np.where(sig == 2)[0][-1]
        
        # その後の空白(Inter-char/Word)の中で Boundary=1 を探す
        found_boundary = False
        # Dah(2) が終わった直後のフレーム（空白の開始）を特定
        space_start_idx = last_dah_idx + 1
    
        # 空白区間をスキャンして Boundary=1 を探す
        for i in range(space_start_idx, len(sig)):
            if bound[i] == 1.0:
                elapsed_frames = i - space_start_idx
                # 180ms = 18フレーム。
                # 物理的な定義: 空白開始から 3ユニット(180ms)経過した瞬間。
                # 許容範囲を 15〜22 フレームとする (サンプリング誤差考慮)
                self.assertTrue(15 <= elapsed_frames <= 25, f"Boundary timing error: {elapsed_frames} frames from space start (Expected ~18)")
                # 境界は Inter-char (4), Inter-word (5) または 背景 (0) で発生しうる
                self.assertIn(sig[i], [0, 4, 5], f"Boundary in unexpected signal class: {sig[i]}")
                # 信号のON区間(1, 2)や文字内空白(3)と重なっていないことを厳密に確認
                self.assertNotIn(sig[i], [1, 2, 3], f"Boundary overlapped with signal! Class: {sig[i]}")
                found_boundary = True
                break
        
        self.assertTrue(found_boundary, "Boundary label not found after character")

    def test_no_boundary_in_intra_space(self):
        # 文字内の空白(Intra-char space, クラス3)には Boundary が立たないことを確認
        text = "M" # --
        waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=20, snr_db=100)
        
        sig = signal_labels.numpy()
        bound = boundary_labels.numpy()
        
        intra_indices = np.where(sig == 3)[0]
        for idx in intra_indices:
            self.assertEqual(bound[idx], 0.0, "Boundary must NOT exist in intra-character space")

if __name__ == '__main__':
    unittest.main()