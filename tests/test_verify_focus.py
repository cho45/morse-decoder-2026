import sys
import os
import random
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_gen import CWDataset
import config


def test_verify_focus_logic():
    num_samples = 100
    all_chars = "KMRSTLO"
    focus_chars = "TLO"
    dataset = CWDataset(
        num_samples=num_samples,
        allowed_chars=all_chars,
        min_len=10,
        max_len=10,
        focus_chars=focus_chars,
        focus_prob=1.0
    )
    
    total_chars = 0
    char_counts = Counter()
    
    for i in range(num_samples):
        _, label, _, _, _, _ = dataset[i]
        
        tokens = [c for c in label if c != ' ']
        char_counts.update(tokens)
        total_chars += len(tokens)
        
    focus_count = sum(char_counts[c] for c in focus_chars)
    non_focus_count = total_chars - focus_count
    
    focus_ratio = focus_count / total_chars
    
    expected_ratio = 0.5 + 0.5 * (len(focus_chars) / len(all_chars))
    
    assert focus_ratio > 0.6, f"Focus logic not working. Focus ratio: {focus_ratio:.2f}, expected > 0.6"
