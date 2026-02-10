import unittest
import numpy as np
import torch
import config
from config import (
    SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE,
    INTERNAL_ID_INTRA_GAP, INTERNAL_ID_INTER_CHAR
)
from data_gen import MorseGenerator, TimingGenerator

class TestBoundarySafety(unittest.TestCase):
    def setUp(self):
        self.sample_rate = config.SAMPLE_RATE
        self.gen = MorseGenerator(sample_rate=self.sample_rate)
        # 1 unit (dot) のサンプル数 (20 WPM)
        self.dot_samples = int(round((1.2 / 20) * self.sample_rate))

    def test_ensure_char_space_completeness(self):
        """文字間空白が完全に入らない場合、その文字自体を含めないべき"""
        text = "EE" # E(1) + space(3) + E(1)
        wpm = 20
        
        # E(1) = 1 unit
        # space(3) = 3 units
        # E(1) = 1 unit
        # Total needed for "EE" = 5 units
        
        # ケース1: 最初の 'E' + space(3) がギリギリ入る
        # duration = 4 units
        target_samples = self.dot_samples * 4
        max_duration = target_samples / self.sample_rate
        
        timing = self.gen.generate_timing(text, wpm=wpm, max_duration=max_duration)
        
        # 期待: E(1), Space(3) が入る。最後の E は入らない。
        # 現在の実装でもこれは満たすはず (次のEが入らないから)
        self.assertEqual(len(timing), 2, "Should contain E and Space")
        self.assertEqual(timing[0][0], SIG_ID_DIT) # Dit
        self.assertEqual(timing[1][0], INTERNAL_ID_INTER_CHAR) # Char space
        self.assertEqual(timing[1][1], self.dot_samples * 3) # Full space

        # ケース2: 最初の 'E' は入るが、space が 2 units 分しか入らない
        # duration = 3 units
        target_samples = self.dot_samples * 3
        max_duration = target_samples / self.sample_rate
        
        timing = self.gen.generate_timing(text, wpm=wpm, max_duration=max_duration)
        
        # 現在の実装の挙動（予想）: E(1) が入り、Space が 2 units で切れる
        # 理想の挙動: E 自体が入らない（空白が完結しないなら文字も入れない）
        # または、E は入るが Space が切れることを許容しない
        
        # ここでは「文字を入れるなら、その直後の文字間空白（3 units）まで確保できなければならない」という強い制約を課す
        # そうしないとバウンダリラベルが生成されないため。
        
        # もし E だけ入って Space が切れると、その E に対するバウンダリラベルは生成されない（生成条件が Space 完了時のため）
        
        # TimingGenerator のリストを確認
        has_signal = any(t[0] in [SIG_ID_DIT, SIG_ID_DAH] for t in timing)
        has_full_space = False
        if len(timing) >= 2:
            if timing[1][0] == INTERNAL_ID_INTER_CHAR and timing[1][1] == self.dot_samples * 3:
                has_full_space = True
        
        # 信号が含まれているなら、完全なスペースも含まれているべき
        if has_signal:
            self.assertTrue(has_full_space, "If char is included, full char space must also be included")

    def test_boundary_label_generation_on_truncated_space(self):
        """空白が切り詰められた場合、バウンダリラベルが生成されるか確認"""
        text = "E E"
        wpm = 20
        
        # Space が半分しか入らない長さ
        target_samples = int(self.dot_samples * 2.5) # E(1) + 1.5 space
        max_duration = target_samples / self.sample_rate
        
        # 波形とラベルを生成
        waveform, signal_labels, boundary_labels = self.gen.generate_waveform(
            self.gen.generate_timing(text, wpm=wpm, max_duration=max_duration),
            wpm=wpm
        )
        
        # バウンダリラベルが立っているか？
        # LabelGenerator は space終了時にラベルを立てるため、spaceが完結していないと立たないはず
        has_boundary = np.max(boundary_labels) > 0.5
        
        # 修正前の挙動: 文字は入っているのにバウンダリがない -> 不整合データ
        # 修正後の挙動: 文字自体が入らない（波形が無音） -> 整合（何もない）
        
        # 波形に信号(E)があるか確認
        has_signal = np.max(signal_labels) > 0
        
        if has_signal:
            self.assertTrue(has_boundary, "Signal exists but boundary is missing! Incomplete data.")

if __name__ == '__main__':
    unittest.main()
