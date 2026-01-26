import unittest
import torch
import os
import shutil
from train import Trainer
import argparse

class MockArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.epochs = 10
        self.batch_size = 2
        self.lr = 0.001
        self.weight_decay = 0.0
        self.grad_clip = 5.0
        self.d_model = 16
        self.n_head = 2
        self.num_layers = 1
        self.dropout = 0.0
        self.save_dir = "tests/test_checkpoints"
        self.resume = None
        self.curriculum_phase = 0
        self.freeze_encoder = False

class TestAdaptiveCurriculum(unittest.TestCase):
    def setUp(self):
        if os.path.exists("tests/test_checkpoints"):
            shutil.rmtree("tests/test_checkpoints")
        os.makedirs("tests/test_checkpoints")
        self.args = MockArgs()
        self.trainer = Trainer(self.args)

    def tearDown(self):
        if os.path.exists("tests/test_checkpoints"):
            shutil.rmtree("tests/test_checkpoints")

    def test_update_curriculum_progression(self):
        # Initial state
        self.trainer.current_phase = 1
        self.trainer.phases_since_last_advance = 0
        self.trainer.min_epochs_per_phase = 3
        self.trainer.cer_threshold_to_advance = 0.05
        
        # Epoch 1: Good CER, but not enough time
        self.trainer.update_curriculum(val_cer=0.01)
        self.assertEqual(self.trainer.current_phase, 1)
        self.assertEqual(self.trainer.phases_since_last_advance, 1)
        
        # Epoch 2: Good CER, still waiting
        self.trainer.update_curriculum(val_cer=0.01)
        self.assertEqual(self.trainer.current_phase, 1)
        self.assertEqual(self.trainer.phases_since_last_advance, 2)
        
        # Epoch 3: Good CER, time condition met -> Advance
        self.trainer.update_curriculum(val_cer=0.01)
        self.assertEqual(self.trainer.current_phase, 2)
        self.assertEqual(self.trainer.phases_since_last_advance, 0) # Should reset

    def test_update_curriculum_holding(self):
        # Initial state
        self.trainer.current_phase = 1
        self.trainer.phases_since_last_advance = 10 # Long time
        self.trainer.min_epochs_per_phase = 3
        self.trainer.cer_threshold_to_advance = 0.05
        
        # Bad CER -> Should stay
        self.trainer.update_curriculum(val_cer=0.10)
        self.assertEqual(self.trainer.current_phase, 1)
        self.assertEqual(self.trainer.phases_since_last_advance, 11)

    def test_save_and_load_curriculum_state(self):
        # Setup state
        epoch = 1
        train_loss = 0.5
        val_loss = 0.5
        val_cer = 0.1
        self.trainer.current_phase = 2
        self.trainer.phases_since_last_advance = 2
        
        # Save
        self.trainer.save_checkpoint(epoch, train_loss, val_loss, val_cer, 0.1, 0.1, 0.9)
        
        # Reset trainer state to verify load
        self.trainer.current_phase = 1
        self.trainer.phases_since_last_advance = 0
        
        # Load (Simulate what happens in main, but we test load_checkpoint logic if we had it, 
        # or manually verify the file content matches what we expect load logic to do)
        path = os.path.join(self.args.save_dir, f"checkpoint_epoch_{epoch}.pt")
        checkpoint = torch.load(path)
        
        # Verify saved data
        self.assertEqual(checkpoint['curriculum_phase'], 2)
        self.assertEqual(checkpoint['phases_since_last_advance'], 2)
        
        # Verify loading logic (simulated)
        self.trainer.load_checkpoint_state(checkpoint)
        self.assertEqual(self.trainer.current_phase, 2)
        self.assertEqual(self.trainer.phases_since_last_advance, 2)

    def test_load_checkpoint_precedence(self):
        # Case A: Auto mode (args.curriculum_phase = 0) -> Checkpoint wins
        self.args.curriculum_phase = 0
        self.trainer.current_phase = 1 # Initial
        
        checkpoint = {
            'curriculum_phase': 2,
            'phases_since_last_advance': 5
        }
        
        self.trainer.load_checkpoint_state(checkpoint)
        self.assertEqual(self.trainer.current_phase, 2)
        self.assertEqual(self.trainer.phases_since_last_advance, 5)
        
        # Case B: Manual mode (args.curriculum_phase = 5) -> Args wins
        self.args.curriculum_phase = 5
        self.trainer.current_phase = 5 # Initialized from args
        
        checkpoint = {
            'curriculum_phase': 2,
            'phases_since_last_advance': 5
        }
        
        self.trainer.load_checkpoint_state(checkpoint)
        self.assertEqual(self.trainer.current_phase, 5) # Should NOT be 2
        # phases_since_last_advance should probably be reset or kept?
        # If we force phase, we probably want to start counting from 0?
        # Current implementation keeps it. Let's verify current behavior first.
        self.assertEqual(self.trainer.phases_since_last_advance, 5)

if __name__ == '__main__':
    unittest.main()