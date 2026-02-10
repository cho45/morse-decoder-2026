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
        # (class_id, duration)
        timing = [(0, 0.1), (1, 0.06), (3, 0.06), (2, 0.18)]
        waveform, actual_timing = self.gen.generate_waveform(timing)
        
        expected_len = int(sum(t[1] for t in timing) * self.sample_rate)
        self.assertEqual(len(waveform), expected_len)
        self.assertEqual(actual_timing, timing)
        
        self.assertGreater(np.max(np.abs(waveform)), 0.5)

    def test_generate_waveform_types(self):
        """波形タイプのテスト"""
        timing = [(1, 0.1)]
        # WaveformGenerator.generate_waveform no longer takes waveform_type, 
        # it always generates sine as per current implementation.
        # If needed, the implementation should be updated to support types.
        waveform, _ = self.gen.generate_waveform(timing)
        self.assertEqual(len(waveform), int(0.1 * self.sample_rate))
        self.assertGreater(np.max(np.abs(waveform)), 0.5)

    def test_generate_waveform_frequency_drift(self):
        """周波数ドリフトのテスト"""
        timing = [(1, 1.0)]
        wf_no_drift, _ = self.gen.generate_waveform(timing, drift_hz=0.0)
        wf_drift, _ = self.gen.generate_waveform(timing, drift_hz=50.0)
        
        self.assertFalse(np.allclose(wf_no_drift, wf_drift))

    def test_generate_waveform_rise_time(self):
        """ライズタイム（エンベロープ）のテスト"""
        timing = [(1, 0.2)]
        waveform, _ = self.gen.generate_waveform(timing, rise_time=0.05)
        
        self.assertLess(np.abs(waveform[0]), 0.1)
        # 中央付近は大きいはず。ゼロ交差を避けるため周辺の最大値を確認
        mid_idx = int(0.1 * self.sample_rate)
        self.assertGreater(np.max(np.abs(waveform[mid_idx-10:mid_idx+10])), 0.8)

if __name__ == '__main__':
    unittest.main()