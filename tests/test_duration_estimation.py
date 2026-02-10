import unittest
import config
from data_gen import MorseGenerator

class TestDurationEstimation(unittest.TestCase):
    def setUp(self):
        self.sample_rate = config.SAMPLE_RATE
        self.gen = MorseGenerator(sample_rate=self.sample_rate)

    def test_estimate_duration_consistency(self):
        """estimate_duration と generate_timing の結果が一致するか"""
        text = "CQ CQ"
        wpm = 20
        # Farnsworth なしの場合
        estimated_sec = self.gen.estimate_duration(text, wpm=wpm)
        estimated_samples = int(estimated_sec * self.sample_rate)
        
        # padding なしの純粋な長さを取得
        timing = self.gen.generate_timing(text, wpm=wpm, max_duration=None)
        actual_samples = sum(t[1] for t in timing)
        
        # 許容誤差: 1サンプル単位の丸め誤差の蓄積があるため、多少は許容
        diff = abs(estimated_samples - actual_samples)
        self.assertLess(diff, 100, f"Standard WPM: Estimated {estimated_samples}, Actual {actual_samples}, Diff {diff}")

    def test_estimate_duration_farnsworth(self):
        """Farnsworth WPM 使用時の見積もり精度"""
        text = "CQ CQ"
        wpm = 20
        farnsworth_wpm = 10 # 文字間・単語間が長くなる
        
        # generate_timing は farnsworth 対応 (padding なし)
        timing = self.gen.generate_timing(text, wpm=wpm, farnsworth_wpm=farnsworth_wpm, max_duration=None)
        actual_samples = sum(t[1] for t in timing)
        
        # estimate_duration に farnsworth_wpm を渡そうとする
        # 未実装なら TypeError をキャッチして失敗させる（Red）
        try:
            estimated_sec = self.gen.estimate_duration(text, wpm=wpm, farnsworth_wpm=farnsworth_wpm)
        except TypeError:
            self.fail("estimate_duration does not support farnsworth_wpm arg")
            
        estimated_samples = int(estimated_sec * self.sample_rate)
        diff = abs(estimated_samples - actual_samples)
        
        self.assertLess(diff, 100, f"Farnsworth: Estimated {estimated_samples}, Actual {actual_samples}, Diff {diff}")

if __name__ == '__main__':
    unittest.main()
