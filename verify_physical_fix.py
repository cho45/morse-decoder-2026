import torch
import numpy as np
from data_gen import MorseGenerator, generate_sample
import config

def verify():
    gen = MorseGenerator(sample_rate=config.SAMPLE_RATE)
    # 'CQ' is '-.-. --.-'
    # There is a space between '-.-.' and '--.-'
    text = "CQ"
    wpm = 20
    dot_len = 1.2 / wpm
    
    timing = gen.generate_timing(text, wpm=wpm)
    
    print(f"Verifying physical timing for '{text}':")
    has_internal_space = False
    for is_on, duration in timing:
        status = "ON " if is_on else "OFF"
        units = duration / (1.2 / wpm)
        print(f"  {status} | {duration:.4f}s ({units:.1f} units)")
        
        # Check if there's an OFF period of ~3 units (char space) inside 'CQ'
        if not is_on and abs(units - 3.0) < 0.1:
            has_internal_space = True
            
    if has_internal_space:
        print("\nSUCCESS: Internal space correctly detected as OFF period.")
    else:
        print("\nFAILURE: No internal space detected. It might still be treated as ON (dash).")

if __name__ == "__main__":
    verify()