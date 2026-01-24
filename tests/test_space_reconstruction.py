import pytest
import numpy as np

def reconstruct_with_spaces(indices, positions, id_to_char):
    """
    WPM情報に頼らず、文字間の相対的なギャップからスペースを復元するロジック。
    """
    if len(indices) < 2:
        return "".join([id_to_char.get(idx, "") for idx in indices])

    gaps = []
    for j in range(1, len(positions)):
        gaps.append(positions[j] - positions[j-1])
    
    # 文字間隔（3ユニット）と単語間隔（7ユニット）の境界を
    # ギャップの中央値の1.8倍（3 * 1.8 = 5.4）付近に設定。
    # これにより、WPMが未知でも統計的にスペースを識別可能。
    median_gap = np.median(gaps)
    threshold = median_gap * 1.8
    
    hypothesis = ""
    for j in range(len(indices)):
        char = id_to_char.get(indices[j], "")
        if j > 0:
            gap = positions[j] - positions[j-1]
            if gap > threshold:
                hypothesis += " "
        hypothesis += char
    return hypothesis

def test_reconstruct_spaces_standard_speed():
    id_to_char = {1: 'A', 2: 'B', 3: 'C'}
    # 20 WPM想定: 文字間ギャップ ~10, 単語間ギャップ ~25
    indices = [1, 2, 3, 1, 2]
    positions = [10, 20, 30, 55, 65] # "ABC AB"
    
    result = reconstruct_with_spaces(indices, positions, id_to_char)
    assert result == "ABC AB"

def test_reconstruct_spaces_slow_speed():
    id_to_char = {1: 'A', 2: 'B'}
    # 10 WPM想定: ギャップが全体的に大きい
    indices = [1, 1, 2, 2]
    positions = [100, 200, 450, 550] # "AA BB"
    
    result = reconstruct_with_spaces(indices, positions, id_to_char)
    assert result == "AA BB"

def test_reconstruct_spaces_fast_speed():
    id_to_char = {1: 'E', 2: 'T'}
    # 40 WPM想定: ギャップが全体的に小さい
    indices = [1, 2, 1, 2]
    positions = [5, 10, 22, 27] # "ET ET"
    
    result = reconstruct_with_spaces(indices, positions, id_to_char)
    assert result == "ET ET"

def test_reconstruct_spaces_no_spaces():
    id_to_char = {1: 'H', 2: 'I'}
    indices = [1, 2]
    positions = [10, 20] # "HI"
    
    result = reconstruct_with_spaces(indices, positions, id_to_char)
    assert result == "HI"