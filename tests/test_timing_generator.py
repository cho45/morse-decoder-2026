"""
TimingGenerator クラスのユニットテスト
"""
import unittest
import random
import numpy as np
from data_gen import TimingGenerator, MorseEncoder
import config


class TestTimingGenerator(unittest.TestCase):
    def setUp(self):
        self.encoder = MorseEncoder()
        self.timing_gen = TimingGenerator(encoder=self.encoder)
    
    def test_generate_timing_basic(self):
        """基本文字のタイミング生成"""
        text = "A"
        # padding を避けるため、推定される持続時間を max_duration に指定
        tokens = self.encoder.text_to_tokens(text)
        duration = self.timing_gen.estimate_duration(tokens, wpm=20)
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=duration)
        
        # A: .- = (1, 0.06), (3, 0.06), (2, 0.18)
        self.assertEqual(len(timing), 3)
        self.assertEqual(timing[0][0], 1)  # Dit
        self.assertEqual(timing[1][0], 3)  # Intra-char space
        self.assertEqual(timing[2][0], 2)  # Dah

    def test_generate_timing_with_pre_silence(self):
        """pre_silence 引数の動作確認"""
        text = "A"
        pre_silence = 0.5
        timing = self.timing_gen.generate_timing(text, wpm=20, pre_silence=pre_silence)
        
        # 先頭に (0, 0.5) が追加されていること
        self.assertEqual(timing[0], (0, 0.5))
        # その後に A の信号が続くこと
        self.assertEqual(timing[1][0], 1)
    
    def test_generate_timing_with_spaces(self):
        """スペースを含むテキストのタイミング生成"""
        text = "A B"
        timing = self.timing_gen.generate_timing(text, wpm=20)
        
        self.assertGreater(len(timing), 8)
        has_word_space = any(t[0] == 5 for t in timing)
        self.assertTrue(has_word_space, "Inter-word space should exist")
    
    def test_generate_timing_with_jitter(self):
        """jitterパラメータの動作確認"""
        text = "A"
        random.seed(42)
        timing_no_jitter = self.timing_gen.generate_timing(text, wpm=20, jitter=0.0)
        
        random.seed(42)
        timing_with_jitter = self.timing_gen.generate_timing(text, wpm=20, jitter=0.1)
        
        self.assertEqual(len(timing_no_jitter), len(timing_with_jitter))
        self.assertNotAlmostEqual(timing_no_jitter[0][1], timing_with_jitter[0][1])
    
    def test_generate_timing_with_weight(self):
        """weightパラメータの動作確認"""
        text = "A"
        timing_weight_1 = self.timing_gen.generate_timing(text, wpm=20, weight=1.0)
        timing_weight_2 = self.timing_gen.generate_timing(text, wpm=20, weight=2.0)
        
        self.assertEqual(len(timing_weight_1), len(timing_weight_2))
        self.assertAlmostEqual(timing_weight_1[0][1] * 2, timing_weight_2[0][1], places=2)
    
    def test_generate_timing_with_farnsworth_wpm(self):
        """Farnsworth WPMパラメータの動作確認"""
        text = "A B"
        timing_standard = self.timing_gen.generate_timing(text, wpm=20, farnsworth_wpm=20)
        timing_slow = self.timing_gen.generate_timing(text, wpm=20, farnsworth_wpm=10)
        
        standard_word_space = [t for t in timing_standard if t[0] == 5]
        slow_word_space = [t for t in timing_slow if t[0] == 5]
        
        if standard_word_space and slow_word_space:
            self.assertGreater(slow_word_space[0][1], standard_word_space[0][1])
    
    def test_generate_timing_empty_text(self):
        """空文字列の処理"""
        text = ""
        max_duration = 1.0
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=max_duration)
        # 空文字列でも max_duration 分の silence (class 0) が返る
        self.assertEqual(len(timing), 1)
        self.assertEqual(timing[0], (0, max_duration))
    
    def test_generate_timing_with_long_gap(self):
        """Long Gapトークンの処理"""
        text = "A<GAP:0.5>B"
        timing = self.timing_gen.generate_timing(text, wpm=20)
        
        has_long_gap = any(t[0] == 0 for t in timing)
        self.assertTrue(has_long_gap, "Long gap should exist")
        
        long_gap = [t for t in timing if t[0] == 0]
        if long_gap:
            self.assertAlmostEqual(long_gap[0][1], 0.5, places=2)
    
    def test_estimate_wpm_for_target_frames(self):
        """WPM推定のテスト"""
        text = "A B C"
        target_frames = 1000
        tokens = self.encoder.text_to_tokens(text)
        
        wpm = self.timing_gen.estimate_wpm_for_target_frames(tokens, target_frames)
        self.assertGreater(wpm, 0)
        
        # 手計算検証
        target_sec = target_frames * config.HOP_LENGTH / config.SAMPLE_RATE
        total_units = self.encoder.count_units_in_tokens(tokens)
        expected_wpm = int(np.clip((1.2 * total_units) / target_sec, 10, 45))
        self.assertEqual(wpm, expected_wpm)
    
    def test_estimate_wpm_for_target_frames_bounds(self):
        """WPM推定の境界値テスト"""
        text = "A"
        tokens = self.encoder.text_to_tokens(text)
        wpm_min = self.timing_gen.estimate_wpm_for_target_frames(tokens, target_frames=1000, min_wpm=10, max_wpm=20)
        wpm_max = self.timing_gen.estimate_wpm_for_target_frames(tokens, target_frames=1000, min_wpm=30, max_wpm=45)
        
        self.assertGreaterEqual(wpm_min, 10)
        self.assertLessEqual(wpm_min, 20)
        self.assertGreaterEqual(wpm_max, 30)
        self.assertLessEqual(wpm_max, 45)
    
    def test_estimate_max_tokens_for_wpm(self):
        """最大トークン数推定のテスト"""
        wpm = 20
        target_frames = 1000
        max_tokens = self.timing_gen.estimate_max_tokens_for_wpm(wpm, target_frames)
        
        self.assertGreater(max_tokens, 0)
        
        target_sec = max(0.5, (target_frames * config.HOP_LENGTH / config.SAMPLE_RATE) - 0.2)
        total_units = (target_sec * wpm) / 1.2
        expected_max_tokens = int(total_units / 13)
        self.assertEqual(max_tokens, expected_max_tokens)
    
    def test_estimate_duration(self):
        """持続時間推定のテスト"""
        text = "A B"
        wpm = 20
        tokens = self.encoder.text_to_tokens(text)
        duration = self.timing_gen.estimate_duration(tokens, wpm)
        
        self.assertGreater(duration, 0)
        total_units = self.encoder.count_units_in_tokens(tokens)
        expected_duration = total_units * (1.2 / wpm)
        self.assertAlmostEqual(duration, expected_duration, places=2)


    def test_generate_timing_max_duration_padding(self):
        """max_duration に満たない場合、post_silence で埋められるか"""
        text = "A"
        max_duration = 1.0
        # まだ引数がないので、ここで TypeError が出るはず (RED)
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=max_duration)
        
        total_duration = sum(t[1] for t in timing)
        self.assertAlmostEqual(total_duration, max_duration, places=5)
        # 最後が class_id=0 (silence) であること
        self.assertEqual(timing[-1][0], 0)

    def test_generate_timing_max_duration_truncation(self):
        """max_duration を超える場合、適切に打ち切られるか"""
        text = "CQ CQ CQ CQ CQ CQ CQ CQ CQ CQ " # 十分に長いテキスト
        max_duration = 0.5
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=max_duration)
        
        total_duration = sum(t[1] for t in timing)
        self.assertAlmostEqual(total_duration, max_duration, places=5)

    def test_generate_timing_total_duration_complex(self):
        """複雑な条件下でも合計時間が一致するか"""
        text = "CQ DE JA1ABC"
        max_duration = 5.0
        pre_silence = 0.5
        timing = self.timing_gen.generate_timing(text, wpm=25, pre_silence=pre_silence, 
                                               max_duration=max_duration, jitter=0.1)
        
        total_duration = sum(t[1] for t in timing)
        self.assertAlmostEqual(total_duration, max_duration, places=5)
        self.assertEqual(timing[0], (0, pre_silence))

if __name__ == '__main__':
    unittest.main()