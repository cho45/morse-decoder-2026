import torch
import numpy as np
import pytest
from data_gen import CWDataset, MorseGenerator
from curriculum import CurriculumManager
import config

def test_curriculum_order():
    cm = CurriculumManager()
    # 最初のフェーズが数字 '1' とそのプレフィックスを含んでいるか確認
    p1 = cm.get_phase(1)
    print(f"Phase 1 chars: {p1.chars}")
    assert '1' in p1.chars
    # 1 (.----) のプレフィックスは E(.), A(.-), W(.--), J(.---)
    for c in "EAWJ":
        assert c in p1.chars

def test_fixed_duration_10s():
    dataset = CWDataset(num_samples=5, min_wpm=20, max_wpm=20)
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[0]
    
    # サンプルレート 16000 で 10秒なら 160000 サンプル
    expected_samples = 10 * config.SAMPLE_RATE
    assert waveform.shape[0] == expected_samples
    print(f"Waveform shape: {waveform.shape}")

def test_wpm_auto_adjust():
    # 非常に長いテキストを生成して WPM が上がるか確認する
    # CWDataset の内部ロジックをシミュレート
    gen = MorseGenerator()
    # 10秒に収まりにくい長いテキスト
    long_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    
    # 10秒 (160000 samples) に収まる WPM を推定
    target_frames = int(10.0 * 0.9 * config.SAMPLE_RATE / config.HOP_LENGTH)
    wpm = gen.estimate_wpm_for_target_frames(long_text, target_frames=target_frames, min_wpm=10, max_wpm=50)
    
    print(f"Estimated WPM for long text: {wpm}")
    assert wpm > 10 # 初期値より上がっているはず

    timing = gen.generate_timing(long_text, wpm=wpm)
    duration = sum(t[1] for t in timing)
    print(f"Duration at {wpm} WPM: {duration:.2f}s")
    assert duration < 10.0

if __name__ == "__main__":
    test_curriculum_order()
    test_fixed_duration_10s()
    test_wpm_auto_adjust()