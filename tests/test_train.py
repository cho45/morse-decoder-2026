import pytest
import torch
import os
import shutil
import argparse
from train import Trainer
import config

class DummyArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.epochs = 1
        self.batch_size = 2
        self.lr = 3e-4
        self.weight_decay = 1e-4
        self.grad_clip = 5.0
        self.save_dir = "test_checkpoints"
        self.curriculum_phase = 0
        self.resume = None
        self.freeze_encoder = False

def test_trainer_init():
    args = DummyArgs()
    trainer = Trainer(args)
    assert trainer.model is not None
    assert trainer.train_dataset.num_samples == 10

def test_train_epoch():
    args = DummyArgs()
    trainer = Trainer(args)
    # Run a single train epoch with very small data
    avg_loss = trainer.train_epoch(1)
    assert isinstance(avg_loss, float)

def test_validate():
    args = DummyArgs()
    trainer = Trainer(args)
    avg_loss, avg_cer = trainer.validate(1)
    assert isinstance(avg_loss, float)
    assert isinstance(avg_cer, float)

def test_save_checkpoint():
    args = DummyArgs()
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    
    trainer = Trainer(args)
    trainer.save_checkpoint(1, 0.5)
    
    checkpoint_path = os.path.join(args.save_dir, "checkpoint_epoch_1.pt")
    assert os.path.exists(checkpoint_path)
    
    # Clean up
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

def test_collate_fn():
    args = DummyArgs()
    trainer = Trainer(args)
    
    # Create a small dummy batch
    waveform = torch.randn(16000) # 1 second
    batch = [(waveform, "TEST", 20)]
    
    waveforms, targets, lengths, target_lengths, texts, wpms = trainer.collate_fn(batch)
    
    assert waveforms.ndim == 2 # (B, T)
    assert waveforms.size(0) == 1
    assert targets.ndim == 1
    assert texts[0] == "TEST"