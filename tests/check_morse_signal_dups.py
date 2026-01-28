import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen import MORSE_DICT

def check_signal_dups():
    signal_to_chars = {}
    
    for char, signal in MORSE_DICT.items():
        if signal == " ": # スペースは除外
            continue
        if signal not in signal_to_chars:
            signal_to_chars[signal] = []
        signal_to_chars[signal].append(char)
    
    dups_found = False
    print("Checking for duplicate Morse signals in MORSE_DICT...")
    for signal, chars in signal_to_chars.items():
        if len(chars) > 1:
            print(f"Duplicate signal found: {signal}")
            print(f"  Mapped to characters: {', '.join(map(repr, chars))}")
            dups_found = True
            
    if not dups_found:
        print("No duplicate signals found.")
    
    return dups_found

if __name__ == "__main__":
    check_signal_dups()