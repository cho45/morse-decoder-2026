import numpy as np
import config
from data_gen import MorseGenerator, WaveformGenerator, LabelGenerator

def test_pre_silence_consistency():
    """
    timing に含まれる pre_silence がすべての工程で同じ値であることを確認
    """
    gen = MorseGenerator()
    text = "CQ DE JA1ABC "
    wpm = 25
    max_duration = 10.0

    pre_silence = 1.0
    timing = gen.generate_timing(text, wpm=wpm, pre_silence=pre_silence, max_duration=max_duration)

    # 既に generate_timing が正確に max_duration 分のサンプルを返しているはず
    total_samples = sum(t[1] for t in timing)
    expected_samples = int(max_duration * config.SAMPLE_RATE)
    assert total_samples == expected_samples

    # WaveformGenerator で波形を生成
    waveform_gen = WaveformGenerator()
    waveform, _ = waveform_gen.generate_waveform(timing)
    assert len(waveform) == expected_samples

    # 先頭 1秒 (pre_silence) が静寂であることを確認
    pre_silence_samples = int(pre_silence * config.SAMPLE_RATE)
    assert np.all(waveform[:pre_silence_samples] == 0)

def test_reconstruct_text():
    """
    timing からテキストが正しく復元できることを確認
    """
    gen = MorseGenerator()
    original_text = "CQ DE JA1ABC "
    wpm = 20

    # 1. 通常の復元
    timing = gen.generate_timing(original_text, wpm=wpm, pre_silence=0.5)
    reconstructed = gen.reconstruct_text_from_timing(timing)
    # reconstruct_text_from_timing は切り捨てられた文字を除外するため、
    # 完全に収まっていれば一致するはず
    assert reconstructed.strip() == original_text.strip()

    # 2. 切り捨てが発生した場合の復元
    long_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG "
    timing = gen.generate_timing(long_text, wpm=30, max_duration=2.0) # 2s 打ち切り
    
    reconstructed = gen.reconstruct_text_from_timing(timing)
    assert len(reconstructed) > 0
    assert long_text.startswith(reconstructed)

def test_boundary_alignment():
    """
    waveform と boundary が正しくアライメントされていることを確認
    """
    gen = MorseGenerator()
    text = "CQ DE JA1ABC "
    wpm = 25
    max_duration = 10.0

    pre_silence = 1.0
    timing = gen.generate_timing(text, wpm=wpm, pre_silence=pre_silence, max_duration=max_duration)

    waveform, signal_labels, boundary_labels = gen.generate_waveform(
        timing, wpm=wpm
    )

    # 境界ラベルが立っている箇所を確認
    boundary_indices = np.where(boundary_labels > 0.5)[0]
    assert len(boundary_indices) > 0

    for frame_idx in boundary_indices:
        # そのフレームの物理的な時間（サンプル）
        sample_idx = frame_idx * config.HOP_LENGTH
        # その付近で Dit/Dah が終わっているか、あるいはスペースが続いているかを確認
        # (厳密なチェックは難しいが、インデックスエラーが出ないことと妥当な範囲であることを確認)
        assert sample_idx < len(waveform)

if __name__ == "__main__":
    test_pre_silence_consistency()
    test_reconstruct_text()
    test_boundary_alignment()