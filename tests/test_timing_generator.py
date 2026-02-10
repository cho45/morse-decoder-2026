import unittest
import config
from data_gen import TimingGenerator, MorseEncoder

class TestTimingGenerator(unittest.TestCase):
    def setUp(self):
        self.sample_rate = config.SAMPLE_RATE
        self.encoder = MorseEncoder()
        self.timing_gen = TimingGenerator(sample_rate=self.sample_rate, encoder=self.encoder)

    def test_generate_timing_with_pre_silence(self):
        """pre_silence 引数の動作確認"""
        text = "A"
        pre_silence = 0.5
        timing = self.timing_gen.generate_timing(text, wpm=20, pre_silence=pre_silence)
    
        # 先頭に silence samples が追加されていること
        expected_pre_samples = int(pre_silence * self.sample_rate)
        self.assertEqual(timing[0], (0, expected_pre_samples))

    def test_generate_timing_empty_text(self):
        """空文字列の処理"""
        text = ""
        max_duration = 1.0
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=max_duration)
        # 空文字列でも max_duration 分の silence (class 0) が返る
        expected_samples = int(max_duration * self.sample_rate)
        self.assertEqual(len(timing), 1)
        self.assertEqual(timing[0], (0, expected_samples))

    def test_generate_timing_with_long_gap(self):
        """Long Gapトークンの処理"""
        text = "A<GAP:0.5>B"
        timing = self.timing_gen.generate_timing(text, wpm=20)
    
        has_long_gap = any(t[0] == 0 for t in timing)
        self.assertTrue(has_long_gap, "Long gap should exist")
    
        long_gap = [t for t in timing if t[0] == 0]
        if long_gap:
            expected_gap_samples = int(0.5 * self.sample_rate)
            self.assertEqual(long_gap[0][1], expected_gap_samples)

    def test_generate_timing_max_duration_padding(self):
        """max_duration に満たない場合、post_silence で埋められるか"""
        text = "A"
        max_duration = 1.0
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=max_duration)
    
        total_samples = sum(t[1] for t in timing)
        expected_samples = int(max_duration * self.sample_rate)
        self.assertEqual(total_samples, expected_samples)

    def test_generate_timing_max_duration_truncation(self):
        """max_duration を超える場合、適切に打ち切られるか"""
        text = "CQ CQ CQ CQ CQ CQ CQ CQ CQ CQ " # 十分に長いテキスト
        max_duration = 0.5
        timing = self.timing_gen.generate_timing(text, wpm=20, max_duration=max_duration)
    
        total_samples = sum(t[1] for t in timing)
        expected_samples = int(max_duration * self.sample_rate)
        self.assertEqual(total_samples, expected_samples)

    def test_generate_timing_total_duration_complex(self):
        """複雑な条件下でも合計時間が一致するか"""
        text = "CQ DE JA1ABC"
        max_duration = 5.0
        pre_silence = 0.5
        timing = self.timing_gen.generate_timing(text, wpm=25, pre_silence=pre_silence,
                                               max_duration=max_duration, jitter=0.1)
    
        total_samples = sum(t[1] for t in timing)
        expected_samples = int(max_duration * self.sample_rate)
        self.assertEqual(total_samples, expected_samples)

if __name__ == '__main__':
    unittest.main()
