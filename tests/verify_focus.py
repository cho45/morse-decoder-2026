import sys
import os
import random
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_gen import CWDataset
import config

def verify_focus_logic():
    print("Verifying Focus Logic in CWDataset...")
    
    # Setup
    num_samples = 100
    all_chars = "KMRSTLO"
    focus_chars = "TLO"
    dataset = CWDataset(
        num_samples=num_samples,
        allowed_chars=all_chars,
        min_len=10,
        max_len=10,
        focus_chars=focus_chars,
        focus_prob=1.0 # Force focus logic to trigger
    )
    
    # Generate samples and count characters
    total_chars = 0
    char_counts = Counter()
    
    for i in range(num_samples):
        # We don't need waveform, just label
        # But dataset[i] generates waveform too, which is slow.
        # Let's bypass generate_sample and just test the token selection logic if possible.
        # But token selection is inside __getitem__.
        # We'll just run it, 100 samples is fast enough for CPU.
        _, label, _, _ = dataset[i]
        
        # Label is string like "K M R T L O"
        # Split by space or just count non-space chars
        tokens = [c for c in label if c != ' ']
        char_counts.update(tokens)
        total_chars += len(tokens)
        
    print(f"Total characters generated: {total_chars}")
    print("Character Counts:")
    for char, count in char_counts.most_common():
        print(f"  {char}: {count} ({count/total_chars*100:.1f}%)")
        
    # Analyze Focus vs Non-Focus
    focus_count = sum(char_counts[c] for c in focus_chars)
    non_focus_count = total_chars - focus_count
    
    focus_ratio = focus_count / total_chars
    print(f"\nFocus Characters ({focus_chars}): {focus_count} ({focus_ratio*100:.1f}%)")
    print(f"Other Characters (KMRS): {non_focus_count} ({(1-focus_ratio)*100:.1f}%)")
    
    # Expected: Focus chars should be significantly more than 50% because:
    # 1. We force at least 50% to be focus chars.
    # 2. The remaining 50% are random, which ALSO include focus chars.
    # So expected ratio is 0.5 + 0.5 * (len(focus)/len(all))
    # 0.5 + 0.5 * (3/7) = 0.5 + 0.21 = 0.71 (71%)
    
    expected_ratio = 0.5 + 0.5 * (len(focus_chars) / len(all_chars))
    print(f"Expected Ratio (approx): {expected_ratio*100:.1f}%")
    
    if focus_ratio > 0.6: # Allow some variance
        print("\nSUCCESS: Focus logic is working. New characters are over-represented.")
    else:
        print("\nFAILURE: Focus logic is NOT working effectively.")

if __name__ == "__main__":
    verify_focus_logic()