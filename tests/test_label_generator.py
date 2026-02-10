import unittest
import numpy as np
import config
from data_gen import LabelGenerator

class TestLabelGenerator(unittest.TestCase):
    def setUp(self):
        self.sample_rate = config.SAMPLE_RATE
        self.label_gen = LabelGenerator(sample_rate=self.sample_rate)

    def test_generate_signal_frames_basic(self):
        """基本シグナルフレーム生成のテスト"""
        # 簡単なタイミング: Dit, Intra-char space, Dah
        pre_silence = 0.2
        timing = [(0, pre_silence), (1, 0.06), (3, 0.06), (2, 0.18)]
        num_frames = 100
        
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
        
        # pre_silence期間 (0.2s) は 0
        pre_silence_samples = int(pre_silence * self.sample_rate)
        # LabelGenerator は center_sample を見ているため、境界は N_FFT // 2 分ズレる
        pre_silence_frames = (pre_silence_samples - config.N_FFT // 2) // config.HOP_LENGTH
        if pre_silence_frames > 0:
            self.assertTrue(np.all(signal_frames[:pre_silence_frames] == 0))
        
        # Dit期間 (0.06s) は 1
        # center_sample が pre_silence 以降、pre_silence + 0.06s 以前のフレーム
        start_frame = int(np.ceil((pre_silence_samples - config.N_FFT // 2) / config.HOP_LENGTH))
        self.assertEqual(signal_frames[start_frame], 1)

    def test_generate_signal_frames_with_inter_word_space(self):
        """Inter-word spaceのテスト"""
        pre_silence = 0.2
        timing = [(0, pre_silence), (1, 0.06), (5, 0.42)]
        num_frames = 100
        
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
        
        # Inter-word spaceの期間を確認 (0.26s 以降)
        start_sample = int(0.26 * self.sample_rate)
        start_frame = int(np.ceil((start_sample - config.N_FFT // 2) / config.HOP_LENGTH))
        self.assertEqual(signal_frames[start_frame], 3)

    def test_generate_signal_frames_background(self):
        """バックグラウンド（空白）のテスト"""
        timing = [(0, 0.2), (0, 0.5)]  # Silence
        num_frames = 100
        
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
        self.assertTrue(np.all(signal_frames == 0))

    def test_generate_signal_frames_empty_timing(self):
        """空タイミングの処理"""
        timing = []
        num_frames = 100
        
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
        self.assertEqual(len(signal_frames), num_frames)
        self.assertTrue(np.all(signal_frames == 0))

    def test_generate_boundary_frames_inter_char_space(self):
        """Inter-char space後の境界ラベルテスト"""
        pre_silence = 0.2
        timing = [(0, pre_silence), (1, 0.06), (4, 0.18)]
        num_frames = 100
        dot_len_sec = 0.06
        
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_len_sec)
        
        trigger_sample = int(0.44 * self.sample_rate)
        trigger_frame = trigger_sample // config.HOP_LENGTH
        self.assertEqual(boundary_frames[trigger_frame], 1.0)

    def test_generate_boundary_frames_inter_word_space(self):
        """Inter-word space後の境界ラベルテスト"""
        pre_silence = 0.2
        timing = [(0, pre_silence), (1, 0.06), (5, 0.42)]
        num_frames = 100
        dot_len_sec = 0.06
        
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_len_sec)
        
        # 1. 文字終了境界: 0.44s
        trigger1_sample = int(0.44 * self.sample_rate)
        trigger1_frame = trigger1_sample // config.HOP_LENGTH
        self.assertEqual(boundary_frames[trigger1_frame], 1.0)
        
        # 2. スペース終了境界: 0.68s
        trigger2_sample = int(0.68 * self.sample_rate)
        trigger2_frame = trigger2_sample // config.HOP_LENGTH
        self.assertEqual(boundary_frames[trigger2_frame], 1.0)

    def test_generate_boundary_frames_long_gap(self):
        """Long Gapの境界ラベルテスト"""
        timing = [(0, 0.2), (1, 0.06), (0, 0.5)]
        num_frames = 100
        dot_len_sec = 0.06
        
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_len_sec)
        self.assertTrue(np.all(boundary_frames == 0))

    def test_generate_boundary_frames_pre_silence_consideration(self):
        """pre_silenceを考慮した境界位置テスト"""
        pre_silence = 0.5
        timing = [(0, pre_silence), (1, 0.06), (4, 0.18)]
        num_frames = 100
        dot_len_sec = 0.06
        
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_len_sec)
        
        trigger_sample = int(0.74 * self.sample_rate)
        trigger_frame = trigger_sample // config.HOP_LENGTH
        self.assertEqual(boundary_frames[trigger_frame], 1.0)

    def test_boundary_frame_range(self):
        """境界ラベルのフレーム範囲テスト"""
        timing = [(0, 0.2), (1, 0.06), (4, 0.18)]
        num_frames = 100
        dot_len_sec = 0.06
        
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_len_sec)
        
        trigger_sample = int(0.44 * self.sample_rate)
        trigger_frame = trigger_sample // config.HOP_LENGTH
        
        for i in range(5):
            if trigger_frame + i < num_frames:
                self.assertEqual(boundary_frames[trigger_frame + i], 1.0)

if __name__ == '__main__':
    unittest.main()
