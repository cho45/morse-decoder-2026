import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from data_gen import MORSE_DICT
from train import Trainer


def test_vocabulary_consistency():
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
    
    for char in list(MORSE_DICT.keys()) + [' ']:
        mock_waveform = torch.zeros(16000)
        mock_signal_labels = torch.zeros(100)
        mock_boundary_labels = torch.zeros(100)
        
        batch = [(mock_waveform, char, 20, mock_signal_labels, mock_boundary_labels, False)]
        
        try:
            _, labels, _, label_lengths, _, _, _, _, _ = trainer.collate_fn(batch)
            
            if label_lengths.item() == 0:
                missing_chars.append(char)
        except Exception as e:
            missing_chars.append(char)
    
    assert not missing_chars, f"The following characters are missing from the vocabulary: {missing_chars}"
