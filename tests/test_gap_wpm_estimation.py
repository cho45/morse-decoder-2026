import unittest
import config
from data_gen import MorseGenerator

class TestGapWPMEstimation(unittest.TestCase):
    def setUp(self):
        # MorseGenerator は内部で TimingGenerator と MorseEncoder を持つ
        self.gen = MorseGenerator()

    def test_wpm_estimation_accounting_for_gap(self):
        """GAPが含まれる場合、WPMの見積もりが正味の時間で行われることを検証"""
        text_no_gap = "PARIS PARIS" # 符号のみ
        text_with_gap = "PARIS <GAP:5.0> PARIS" # 5秒の固定GAPを含む
        
        # 1. GAPなしで 10秒に収める場合の WPM を基準とする
        target_frames_10s = int(10.0 * config.SAMPLE_RATE / config.HOP_LENGTH)
        wpm_base = self.gen.estimate_wpm_for_target_frames(text_no_gap, target_frames=target_frames_10s)
        
        # 2. 5秒の GAP ありで 15秒に収める場合
        # 正味時間は 15s - 5s = 10s となり、1 と同じになるはず
        # したがって計算される WPM も wpm_base と一致すべき
        target_frames_15s = int(15.0 * config.SAMPLE_RATE / config.HOP_LENGTH)
        wpm_with_gap = self.gen.estimate_wpm_for_target_frames(text_with_gap, target_frames=target_frames_15s)
        
        # 許容誤差 1 WPM 以内で一致することを確認
        self.assertAlmostEqual(wpm_with_gap, wpm_base, delta=1)

    def test_wpm_estimation_increase_on_small_net_time(self):
        """GAPによって正味の時間が減った場合、WPMが適切に上昇するか検証"""
        text = "E <GAP:4.0> E" # 4秒の固定時間がある
        
        # 10s目標 (正味6s) の WPM
        wpm_10s = self.gen.estimate_wpm_for_target_frames(text, target_frames=int(10.0 * config.SAMPLE_RATE / config.HOP_LENGTH))
        
        # 5s目標 (正味1s) の WPM
        # 使える正味時間が 6s -> 1s に激減するため、WPM は大幅に上がるはず
        wpm_5s = self.gen.estimate_wpm_for_target_frames(text, target_frames=int(5.0 * config.SAMPLE_RATE / config.HOP_LENGTH))
        
        self.assertGreater(wpm_5s, wpm_10s, "WPM must increase when net time decreases due to GAP")

if __name__ == '__main__':
    unittest.main()
