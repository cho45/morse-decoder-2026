import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen import MORSE_DICT


def test_check_morse_signal_dups():
    signal_to_chars = {}
    
    for char, signal in MORSE_DICT.items():
        if signal == " ":
            continue
        if signal not in signal_to_chars:
            signal_to_chars[signal] = []
        signal_to_chars[signal].append(char)
    
    dups = []
    for signal, chars in signal_to_chars.items():
        if len(chars) > 1:
            dups.append((signal, chars))
    
    # Check that known duplicates are intentional (e.g., <KA> and <CT> both mean "attention")
    if dups:
        # Verify these are expected duplicates
        known_duplicate_pairs = {
            ('<KA>', '<CT>'): '-.-.-',
        }
        
        for signal, chars in dups:
            chars_tuple = tuple(sorted(chars))
            assert chars_tuple in known_duplicate_pairs, f"Unexpected duplicate signal: {signal} for {chars}"
            assert signal == known_duplicate_pairs[chars_tuple], f"Signal mismatch for duplicate chars {chars_tuple}"
