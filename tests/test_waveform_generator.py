import unittest
import numpy as np
import config
from data_gen import WaveformGenerator

class TestWaveformGenerator(unittest.TestCase):
    def setUp(self):
        self.sample_rate = config.SAMPLE_RATE
        self.gen = WaveformGenerator(sample_rate=self.sample_rate)

    def test_generate_waveform_basic(self):
        """基本波形生成のテスト"""
        # (class_id, samples)
        timing = [
            (0, int(0.1 * self.sample_rate)),
            (1, int(0.06 * self.sample_rate)),
            (3, int(0.06 * self.sample_rate)),
            (2, int(0.18 * self.sample_rate))
        ]
        waveform, actual_timing = self.gen.generate_waveform(timing)
        
        expected_samples = sum(t[1] for t in timing)
        self.assertEqual(len(waveform), expected_samples)
        self.assertEqual(actual_timing, timing)

    def test_generate_waveform_frequency_drift(self):
        """周波数ドリフトのテスト"""
        timing = [(1, int(1.0 * self.sample_rate))]
        wf_no_drift, _ = self.gen.generate_waveform(timing, drift_hz=0.0)
        wf_drift, _ = self.gen.generate_waveform(timing, drift_hz=50.0)
        
        self.assertEqual(len(wf_no_drift), len(wf_drift))
        # ドリフトがあると波形が異なるはず
        self.assertFalse(np.allclose(wf_no_drift, wf_drift))

    def test_generate_waveform_rise_time(self):
        """ライズタイム（エンベロープ）のテスト"""
        timing = [(1, int(0.2 * self.sample_rate))]
        waveform, _ = self.gen.generate_waveform(timing, rise_time=0.05)
        
        # 最初のサンプルは 0
        self.assertAlmostEqual(waveform[0], 0.0, places=2)
        
        # ライズが終わった後 (0.05s = 800samples 以降) のピークを確認
        # 800サンプルから 2400サンプルの間は定常状態のはず
        steady_state = waveform[1000:2200]
        self.assertAlmostEqual(np.max(np.abs(steady_state)), 1.0, places=1)
        
        # 最後のサンプルも 0
        self.assertAlmostEqual(waveform[-1], 0.0, places=2)

if __name__ == '__main__':
    unittest.main()
