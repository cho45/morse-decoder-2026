import unittest
import numpy as np
import torch
import config
from config import SIG_ID_BLANK
from data_gen import CWDataset

class TestGapInsertion(unittest.TestCase):
    def detect_max_silence_duration(self, signal_labels):
        """signal_labels 内の最大連続無音時間を秒数で返す"""
        # signal_labels は Tensor なので numpy に変換
        labels = signal_labels.numpy() if hasattr(signal_labels, 'numpy') else signal_labels
        
        max_consecutive = 0
        current_consecutive = 0
        for l in labels:
            if l == SIG_ID_BLANK:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # フレームあたりの秒数
        sec_per_frame = config.HOP_LENGTH / config.SAMPLE_RATE
        return max_consecutive * sec_per_frame

    def test_phrase_link_gap_physical(self):
        """フレーズ連結時に 1秒以上の物理的な無音（GAP）が存在することを検証"""
        # フレーズ確率 1.0, GAP確率 1.0
        # フレーズ連結は 1 or 2 なので、何度か試せば 2フレーズ（GAPあり）が生成されるはず
        dataset = CWDataset(phrase_prob=1.0, gap_prob=1.0, num_samples=100)
        
        found_long_gap = False
        for i in range(20):
            item = dataset[i]
            signal_labels = item[3]
            
            # フレーズ連結時の GAP は 1.0-3.0秒。0.9秒以上あれば検知とみなす
            if self.detect_max_silence_duration(signal_labels) >= 0.9:
                found_long_gap = True
                break
        
        self.assertTrue(found_long_gap, "No physical long gap detected in phrase samples")

    def test_random_text_gap_physical(self):
        """ランダム文字列内に 1秒以上の物理的な無音（GAP）が存在することを検証"""
        # ランダム文字列, GAP確率 1.0
        dataset = CWDataset(phrase_prob=0.0, gap_prob=1.0, num_samples=100, min_len=10)
        
        found_long_gap = False
        for i in range(20):
            item = dataset[i]
            signal_labels = item[3]
            
            # ランダム時の GAP は 1.0-5.0秒
            if self.detect_max_silence_duration(signal_labels) >= 0.9:
                found_long_gap = True
                break
        
        self.assertTrue(found_long_gap, "No physical long gap detected in random text samples")

    def test_label_text_no_gap_trash(self):
        """戻り値の label 文字列から <GAP:...> が除去されていることを検証"""
        dataset = CWDataset(gap_prob=1.0, phrase_prob=0.5, num_samples=100)
        
        for i in range(50):
            item = dataset[i]
            label = item[1]
            self.assertNotIn("<GAP:", label, f"GAP token leaked into label text: {label}")
            # Prosigns (e.g. <SK>) は正当な文字なので、それ以外に GAP 特有のパターンがないか確認
            # <GAP: の形式でなければ OK

if __name__ == '__main__':
    unittest.main()