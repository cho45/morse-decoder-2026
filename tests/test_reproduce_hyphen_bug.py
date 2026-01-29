import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from train import Trainer


def test_hyphen_tokenization():
    text = "A-B-C"
    
    class Args:
        lr = 3e-4
        freeze_encoder = False
        save_dir = "checkpoints"
        samples_per_epoch = 1000
        curriculum_phase = 0
        grad_clip = 5.0
    
    args = Args()
    trainer = Trainer(args)
    
    mock_waveform = torch.zeros(16000)
    mock_signal_labels = torch.zeros(100)
    mock_boundary_labels = torch.zeros(100)
    
    batch = [(mock_waveform, text, 20, mock_signal_labels, mock_boundary_labels, False)]
    
    _, labels, _, label_lengths, _, _, _, _, _ = trainer.collate_fn(batch)
    
    expected_length = 5
    assert label_lengths.item() == expected_length, f"Expected length {expected_length}, but got {label_lengths.item()}"
    assert '-' in config.CHAR_TO_ID, "'-' is missing from config.CHAR_TO_ID"
