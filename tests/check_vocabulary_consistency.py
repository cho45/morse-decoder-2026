import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from data_gen import MORSE_DICT
from train import Trainer

def test_vocabulary_consistency():
    print("Testing vocabulary consistency across MORSE_DICT and config.CHARS...")
    
    # Mock args for Trainer
    class Args:
        lr = 3e-4
        freeze_encoder = False
        save_dir = "checkpoints"
        samples_per_epoch = 1000
        curriculum_phase = 0
        grad_clip = 5.0
    
    args = Args()
    trainer = Trainer(args)
    
    missing_chars = []
    
    # Check every character in MORSE_DICT
    for char in MORSE_DICT.keys():
        if char == ' ': # Space is handled separately by Signal Head
            continue
            
        # Try to tokenize this character
        mock_waveform = torch.zeros(16000)
        mock_signal_labels = torch.zeros(100)
        mock_boundary_labels = torch.zeros(100)
        
        # We test single char tokenization
        batch = [(mock_waveform, char, 20, mock_signal_labels, mock_boundary_labels, False)]
        
        try:
            _, labels, _, label_lengths, _, _, _, _, _ = trainer.collate_fn(batch)
            
            if label_lengths.item() == 0:
                missing_chars.append(char)
                print(f"FAILED: Character '{char}' resulted in 0 tokens.")
            else:
                # print(f"PASSED: '{char}' -> token ID {labels.tolist()}")
                pass
        except Exception as e:
            missing_chars.append(char)
            print(f"ERROR: Exception while tokenizing '{char}': {e}")

    if not missing_chars:
        print("\nSUCCESS: All characters in MORSE_DICT are correctly mapped in config.py!")
    else:
        print(f"\nFAILURE: The following characters are missing from the vocabulary: {missing_chars}")
        sys.exit(1)

if __name__ == "__main__":
    test_vocabulary_consistency()