import unittest
import numpy as np
import config
from config import (
    SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE,
    INTERNAL_ID_INTRA_GAP, INTERNAL_ID_INTER_CHAR
)
from data_gen import LabelGenerator

class TestLabelGenerator(unittest.TestCase):
    def setUp(self):
        self.label_gen = LabelGenerator()

    def test_generate_signal_frames(self):
        """シグナルフレームの基本生成テスト (定数化のみ)"""
        # Dit(1600), Gap(1600), Dah(4800)
        timing = [(SIG_ID_DIT, 1600), (INTERNAL_ID_INTRA_GAP, 1600), (SIG_ID_DAH, 4800)]
        num_frames = 100
        
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
        
        # オリジナルの期待値（1600 samples ごとの切り替わり）
        # 1600 / 160 = 10フレーム。
        # ただし DSP のオフセット (512 samples) があるため、正確に 0-6 が DIT、7-16 が BLANK... となる
        self.assertEqual(signal_frames[0], SIG_ID_DIT)
        self.assertEqual(signal_frames[6], SIG_ID_DIT)
        self.assertEqual(signal_frames[7], SIG_ID_BLANK)
        self.assertEqual(signal_frames[16], SIG_ID_BLANK)
        self.assertEqual(signal_frames[17], SIG_ID_DAH)

    def test_generate_signal_frames_with_inter_word_space(self):
        """Inter-word space (SIG_ID_WORD_SPACE) のマッピングテスト"""
        timing = [(SIG_ID_BLANK, 3200), (SIG_ID_DIT, 960), (SIG_ID_WORD_SPACE, 6720)]
        num_frames = 100
    
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
    
        # 3200 + 960 = 4160 samples 以降が WORD_SPACE (3)
        # 25 * 160 + 512 = 4512 (> 4160) なので WORD_SPACE
        self.assertEqual(signal_frames[25], SIG_ID_WORD_SPACE)
        self.assertEqual(signal_frames[30], SIG_ID_WORD_SPACE)

    def test_generate_boundary_frames_inter_char_space(self):
        """Inter-char space 完了時点での境界ラベルテスト (元の仕様)"""
        dot_samples = 960
        # pre(3200) + dot(960) + space(2880) = 7040
        timing = [(SIG_ID_BLANK, 3200), (SIG_ID_DIT, 960), (INTERNAL_ID_INTER_CHAR, 2880)]
        num_frames = 100
    
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_samples)
        
        # 境界位置: 7040 samples (空白の完了時点 = 元の仕様)
        trigger_frame = 7040 // config.HOP_LENGTH
        self.assertEqual(boundary_frames[trigger_frame], 1.0)

    def test_generate_boundary_frames_inter_word_space(self):
        """Inter-word space 内の境界ラベルテスト (元の仕様)"""
        dot_samples = 960
        # pre(3200) + dot(960) + word_space(6720)
        timing = [(SIG_ID_BLANK, 3200), (SIG_ID_DIT, 960), (SIG_ID_WORD_SPACE, 6720)]
        num_frames = 100
    
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_samples)
        
        # 1. 前の文字の終了 (3ユニット分): 3200 + 960 + (3 * 960) = 7040 samples
        frame1 = 7040 // config.HOP_LENGTH
        self.assertEqual(boundary_frames[frame1], 1.0)
        
        # 2. 空白全体の終了: 3200 + 960 + 6720 = 10880 samples
        frame2 = 10880 // config.HOP_LENGTH
        self.assertEqual(boundary_frames[frame2], 1.0)

if __name__ == '__main__':
    unittest.main()