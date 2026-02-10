import torch
import config
from data_gen import CWDataset
import pytest

def test_random_generation_prosign_integrity():
    # プロサインを含むボキャブラリを設定
    chars = "ABC<NJ><SK>"
    dataset = CWDataset(
        num_samples=100,
        min_len=10, 
        allowed_chars=chars,
        phrase_prob=0.0 # ランダム生成を強制
    )
    
    # 大量のサンプルを生成し、ターゲット長が min_len (10) を下回らないかチェック
    for i in range(100):
        # returns waveform, label, wpm, signal_labels, boundary_labels, is_phrase
        sample = dataset[i]
        label = sample[1]
        wpm = sample[2]
        
        actual_len = len(label)
        
        # バグがあれば < や > が無視されて 10 未満になる
        assert actual_len >= 10, f"Sample {i} failed: Len {actual_len} < 10. WPM: {wpm}"

if __name__ == "__main__":
    pytest.main([__file__])