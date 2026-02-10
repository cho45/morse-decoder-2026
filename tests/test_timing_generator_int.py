import unittest
import config
from config import SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE, INTERNAL_ID_INTER_CHAR, INTERNAL_ID_INTRA_GAP
from data_gen import TimingGenerator

class TestTimingGeneratorInt(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.gen = TimingGenerator(sample_rate=self.sample_rate)

    def test_timing_elements_are_integers(self):
        """全てのタイミング要素が整数（サンプル数）であることを確認"""
        wpm = 20
        # generate_timing の戻り値の各要素の duration 部分が int かチェック
        timing = self.gen.generate_timing("TEST", wpm=wpm)
        for class_id, num_samples in timing:
            self.assertIsInstance(num_samples, int, f"Duration should be int (samples), got {type(num_samples)} for class {class_id}")

    def test_total_samples_match_exactly(self):
        """合計サンプル数が max_duration * sample_rate と厳密に一致することを確認"""
        wpm = 20
        max_duration = 0.5 # 0.5s = 8000 samples
        target_samples = int(max_duration * self.sample_rate)
        
        timing = self.gen.generate_timing("A", wpm=wpm, max_duration=max_duration)
        total_samples = sum(num_samples for _, num_samples in timing)
        
        self.assertEqual(total_samples, target_samples, f"Total samples {total_samples} mismatch with target {target_samples}")

    def test_unit_samples_consistency(self):
        """1ユニットのサンプル数が WPM に基づいて正しく計算されているか"""
        wpm = 20 # 1 unit = 1.2 / 20 = 0.06s
        # 0.06s * 16000 = 960 samples
        expected_unit_samples = 960
        
        # 'E' は 1ユニットの Dit
        timing = self.gen.generate_timing("E", wpm=wpm, pre_silence=0)
        # timing: [(SIG_ID_DIT, unit_samples), ...]
        dit_samples = timing[0][1]
        self.assertEqual(dit_samples, expected_unit_samples)

if __name__ == '__main__':
    unittest.main()
