import torch
import torch.nn as nn
from train import Trainer
import config

class DummyArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.epochs = 1
        self.batch_size = 2
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip = 5.0
        self.save_dir = "test_checkpoints"
        self.curriculum_phase = 0
        self.resume = None
        self.freeze_encoder = False

def test_multitask_loss_calculation():
    args = DummyArgs()
    trainer = Trainer(args)
    
    # Dummy data
    batch_size = 2
    T = 50
    num_classes = config.NUM_CLASSES
    
    # Model outputs
    logits = torch.randn(batch_size, T, num_classes).to(trainer.device)
    signal_logits = torch.randn(batch_size, T, config.NUM_SIGNAL_CLASSES).to(trainer.device)
    boundary_logits = torch.randn(batch_size, T, 1).to(trainer.device)
    
    # Targets
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.long).to(trainer.device)
    target_lengths = torch.tensor([2, 2], dtype=torch.long).to(trainer.device)
    input_lengths = torch.tensor([T, T], dtype=torch.long).to(trainer.device)
    
    # Signal targets (multi-class)
    signal_targets = torch.randint(0, config.NUM_SIGNAL_CLASSES, (batch_size, T)).long().to(trainer.device)
    boundary_targets = torch.randint(0, 2, (batch_size, T, 1)).float().to(trainer.device)
    
    # Test CTC Loss part
    logits_t = logits.transpose(0, 1).log_softmax(2)
    ctc_loss = trainer.criterion(logits_t, targets, input_lengths, target_lengths)
    assert not torch.isnan(ctc_loss)
    
    # Test BCE Loss part (using the same logic we will add to train.py)
    bce_criterion = nn.CrossEntropyLoss()
    bce_loss = bce_criterion(signal_logits.transpose(1, 2), signal_targets)
    assert not torch.isnan(bce_loss)
    
    # Total loss
    total_loss = ctc_loss + 0.1 * bce_loss
    assert total_loss > 0