import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import config
from data_gen import MorseGenerator, CWDataset

def test_space_in_chars():
    """CHARS にスペースが含まれていることを確認"""
    assert ' ' in config.CHARS

def test_tokenize_space():
    """スペースがトークンとして認識されることを確認"""
    gen = MorseGenerator()
    tokens = gen.text_to_morse_tokens("CQ CQ ")
    # 期待値: ['C', 'Q', ' ', 'C', 'Q', ' ']
    assert ' ' in tokens
    assert tokens[-1] == ' '

def test_dataset_ends_with_space():
    """Dataset が生成するテキストの末尾に必ずスペースが含まれていることを確認"""
    dataset = CWDataset(num_samples=10)
    for i in range(len(dataset)):
        _, label, _, _, _, _ = dataset[i]
        assert label.endswith(' '), f"Label '{label}' does not end with space"

def test_cer_includes_space():
    """CER の計算にスペースが含まれることを確認"""
    from inference_utils import calculate_cer
    ref = "ABC "
    hyp = "ABC"
    # スペースが含まれる場合、不一致となるはず
    cer = calculate_cer(ref, hyp)
    assert cer > 0