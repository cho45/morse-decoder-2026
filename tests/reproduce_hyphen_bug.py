import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from train import Trainer

def test_hyphen_tokenization():
    print("Testing hyphen tokenization...")
    text = "A-B-C"
    
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
    
    # Simulate the data loading process
    # We'll use a mock batch to test collate_fn
    mock_waveform = torch.zeros(16000)
    mock_signal_labels = torch.zeros(100)
    mock_boundary_labels = torch.zeros(100)
    
    batch = [(mock_waveform, text, 20, mock_signal_labels, mock_boundary_labels, False)]
    
    _, labels, _, label_lengths, _, _, _, _, _ = trainer.collate_fn(batch)
    
    print(f"Original text: {text} (length: {len(text)})")
    print(f"Encoded labels: {labels}")
    print(f"Label length: {label_lengths.item()}")
    
    # "A-B-C" should result in 5 tokens: A, -, B, -, C
    expected_length = 5
    if label_lengths.item() == expected_length:
        print("SUCCESS: Hyphen correctly tokenized.")
    else:
        print(f"FAILURE: Expected length {expected_length}, but got {label_lengths.item()}")
        # Check if '-' is in CHAR_TO_ID
        if '-' not in config.CHAR_TO_ID:
            print("REASON: '-' is missing from config.CHAR_TO_ID")
        sys.exit(1)

if __name__ == "__main__":
    test_hyphen_tokenization()