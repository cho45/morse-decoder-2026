import numpy as np
import random
import config
from data_gen import MorseGenerator, WaveformGenerator, LabelGenerator
import pytest

def test_pre_silence_consistency():
    """
    timing に含まれる pre_silence がすべての工程で同じ値であることを確認
    """
    gen = MorseGenerator()
    text = "CQ DE JA1ABC "
    wpm = 25
    max_duration = 10.0
    
    pre_silence = 1.0
    timing = gen.generate_timing(text, wpm=wpm, pre_silence=pre_silence)
    
    # 最後に不足分の silence (post_silence) を追加して max_duration に合わせる
    total_timing_duration = sum(t[1] for t in timing)
    post_silence = max(0.0, max_duration - total_timing_duration)
    timing = timing + [(0, post_silence)]

    # WaveformGenerator で波形を生成
    waveform_gen = WaveformGenerator()
    waveform, _ = waveform_gen.generate_waveform(timing)
    
    label_gen = LabelGenerator()
    num_frames = (len(waveform) - config.N_FFT) // config.HOP_LENGTH + 1
    signal_frames = label_gen.generate_signal_frames(timing, num_frames)
    
    # 波形の立ち上がり位置を確認
    signal_indices = np.where(np.abs(waveform) > 0.1)[0]
    if len(signal_indices) > 0:
        actual_start_sample = signal_indices[0]
        expected_start_sample = int(pre_silence * config.SAMPLE_RATE)
        rise_offset = 17 
        assert abs(actual_start_sample - (expected_start_sample + rise_offset)) <= 2
        
        # signal_frames の開始位置も確認
        actual_start_frame = np.where(signal_frames > 0)[0][0]
        expected_start_frame = int(np.ceil((expected_start_sample - config.N_FFT // 2) / config.HOP_LENGTH))
        assert actual_start_frame == expected_start_frame

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
    assert reconstructed == original_text

    # 2. 切り捨てが発生した場合の復元
    long_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG "
    timing = gen.generate_timing(long_text, wpm=30)
    
    truncated_timing = []
    current_sum = 0
    for class_id, duration in timing:
        if current_sum + duration > 2.0:
            break
        truncated_timing.append((class_id, duration))
        current_sum += duration
    
    reconstructed = gen.reconstruct_text_from_timing(truncated_timing)
    assert long_text.startswith(reconstructed)
    assert len(reconstructed) > 0

def test_reconstruct_text_with_gap():
    """
    GAPトークンを含むタイミングからの復元テスト
    """
    gen = MorseGenerator()
    # GAPトークンはテキスト復元時には無視される（またはデコードを継続する）
    text = "CQ<GAP:1.0>DE "
    timing = gen.generate_timing(text, wpm=25)
    
    reconstructed = gen.reconstruct_text_from_timing(timing)
    # <GAP:1.0> は TimingGenerator で (0, 1.0) に変換され、
    # reconstruct_text_from_timing では無視されるため、"CQDE " になるはず
    # （注：元の MorseEncoder の text_to_tokens も <GAP> を文字として保持しているが、
    #  MORSE_DICT にはないため、generate_timing では (0, duration) になる）
    assert "CQ" in reconstructed
    assert "DE" in reconstructed

def test_reconstruct_text_edge_cases():
    """
    復元ロジックのエッジケース
    """
    gen = MorseGenerator()
    
    # 1. 空のタイミング
    assert gen.reconstruct_text_from_timing([]) == ""
    
    # 2. 不完全なシンボル（末尾で切れる）
    # A (.-) の . だけ
    dot_len = 0.06
    incomplete_timing = [(1, dot_len)] 
    # . は 'E' なので、'E' が復元されるのが正しい
    assert gen.reconstruct_text_from_timing(incomplete_timing) == "E"
    
    # 3. 未知のモールス符号
    # ...... (6 dots) は MORSE_DICT にない
    unknown_timing = [(1, dot_len), (3, dot_len)] * 5 + [(1, dot_len)]
    # inverse_morse.get("", "") は "" なので、何も追加されない
    assert gen.reconstruct_text_from_timing(unknown_timing) == ""

def test_boundary_alignment():
    """
    waveform と boundary が正しくアライメントされていることを確認
    """
    gen = MorseGenerator()
    text = "CQ DE JA1ABC "
    wpm = 25
    max_duration = 10.0
    
    pre_silence = 1.0
    timing = gen.generate_timing(text, wpm=wpm, pre_silence=pre_silence)
    
    # 最後に不足分の silence (post_silence) を追加
    total_timing_duration = sum(t[1] for t in timing)
    post_silence = max(0.0, max_duration - total_timing_duration)
    timing = timing + [(0, post_silence)]

    waveform, signal_frames, boundary_frames = gen.generate_waveform(
        timing, wpm=wpm
    )
    
    signal_indices = np.where(np.abs(waveform) > 0.1)[0]
    if len(signal_indices) > 0:
        actual_start_sample = signal_indices[0]
        first_boundary_frame = np.where(boundary_frames > 0.5)[0][0]
        
        dot_len_sec = 1.2 / wpm
        dot_len_samples = int(dot_len_sec * config.SAMPLE_RATE)
        # C = 14 units (dah-intra-dit-intra-dah-intra-dit-inter)
        expected_boundary_sample = int(pre_silence * config.SAMPLE_RATE) + 14 * dot_len_samples
        expected_boundary_frame = expected_boundary_sample // config.HOP_LENGTH
        
        assert abs(first_boundary_frame - expected_boundary_frame) <= 1
