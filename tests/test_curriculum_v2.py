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

def test_fixed_duration_target():
    dataset = CWDataset(num_samples=5, min_wpm=20, max_wpm=20)
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[0]
    
    # config.TRAIN_DURATION に一致するか確認
    expected_samples = int(config.TRAIN_DURATION * config.SAMPLE_RATE)
    assert waveform.shape[0] == expected_samples
    print(f"Waveform shape: {waveform.shape}")

def test_wpm_auto_adjust():
    # 長いテキストを生成して WPM が上がるか確認する
    gen = MorseGenerator()
    long_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    
    # 指定した時間 (TRAIN_DURATION) に収まる WPM を推定
    target_frames = int(config.TRAIN_DURATION * 0.9 * config.SAMPLE_RATE / config.HOP_LENGTH)
    wpm = gen.estimate_wpm_for_target_frames(long_text, target_frames=target_frames, min_wpm=10, max_wpm=50)
    
    print(f"Estimated WPM for long text: {wpm}")
    assert wpm > 10 

    timing = gen.generate_timing(long_text, wpm=wpm)
    total_samples = sum(t[1] for t in timing)
    expected_samples = int(config.TRAIN_DURATION * config.SAMPLE_RATE)
    print(f"Total samples at {wpm} WPM: {total_samples}")
    assert total_samples <= expected_samples

if __name__ == "__main__":
    test_curriculum_order()
    test_fixed_duration_10s()
    test_wpm_auto_adjust()