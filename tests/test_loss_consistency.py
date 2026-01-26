import torch
import torch.nn as nn
from train import Trainer
import config
import argparse

class MockArgs:
    def __init__(self):
        self.lr = 3e-4
        self.freeze_encoder = False
        self.samples_per_epoch = 100
        self.save_dir = "checkpoints_test"
        self.curriculum_phase = 1
        self.batch_size = 2
        self.grad_clip = 5.0

def test_train_val_loss_consistency():
    """
    学習モードと検証モードで、境界マスキングを含む損失計算が数学的に一貫しているか検証する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = MockArgs()
    trainer = Trainer(args)
    trainer.model.to(device)
    
    # 固定の入力データを作成
    batch_size = 2
    seq_len = 100
    num_mels = config.N_BINS
    num_classes = config.NUM_CLASSES
    
    mels = torch.randn(batch_size, seq_len, num_mels).to(device)
    input_lengths = torch.full((batch_size,), seq_len // 2, dtype=torch.long).to(device)
    
    targets = torch.tensor([1, 2, 1, 2], dtype=torch.long).to(device)
    target_lengths = torch.tensor([2, 2], dtype=torch.long).to(device)
    
    signal_targets = torch.zeros(batch_size, seq_len // 2, dtype=torch.long).to(device)
    # 境界フラグを一部に立てる
    boundary_targets = torch.zeros(batch_size, seq_len // 2, dtype=torch.float32).to(device)
    boundary_targets[:, 10] = 1.0 
    boundary_targets[:, 30] = 1.0

    # 1. 学習時のロジックをシミュレート (train_epoch 内の計算)
    trainer.model.train()
    (logits, signal_logits, boundary_logits), _ = trainer.model(mels)
    
    # 学習用マスキング
    bound_t = boundary_targets[:, :logits.size(1)]
    is_not_boundary = (bound_t < 0.5)
    logits_masked_train = logits.clone()
    mask_expanded = is_not_boundary.unsqueeze(-1).expand_as(logits[:, :, 1:])
    logits_masked_train[:, :, 1:][mask_expanded] -= 50.0
    
    logits_t_train = logits_masked_train.transpose(0, 1).log_softmax(2)
    ctc_loss_train = trainer.criterion(logits_t_train, targets, input_lengths, target_lengths)

    # 2. 検証時のロジックをシミュレート (validate 内の計算)
    trainer.model.eval()
    with torch.no_grad():
        (logits_val, signal_logits_val, boundary_logits_val), _ = trainer.model(mels)
        
        # 検証用マスキング (train.py の validate メソッドからコピーしたロジック)
        # 予測値ではなく正解ラベルを使用するように修正
        bound_t_val = boundary_targets[:, :logits_val.size(1)]
        is_not_boundary_val = (bound_t_val < 0.5)
        
        logits_masked_val = logits_val.clone()
        mask_expanded_val = is_not_boundary_val.unsqueeze(-1).expand_as(logits_val[:, :, 1:])
        logits_masked_val[:, :, 1:][mask_expanded_val] -= 50.0
        
        logits_t_val = logits_masked_val.transpose(0, 1).log_softmax(2)
        ctc_loss_val = trainer.criterion(logits_t_val, targets, input_lengths, target_lengths)

    # アサーション: ドロップアウトを無効にすれば、Loss はほぼ一致するはず
    # (train モードでも検証用に一時的に dropout を切るか、eval モードで比較)
    print(f"CTC Loss (Train-like path): {ctc_loss_train.item():.6f}")
    print(f"CTC Loss (Val-like path): {ctc_loss_val.item():.6f}")
    
    # 境界外の確率が抑制されているかチェック
    probs_val = torch.softmax(logits_masked_val, dim=-1)
    char_probs_outside = probs_val[:, :, 1:][mask_expanded_val]
    print(f"Max char prob outside boundary: {char_probs_outside.max().item():.6e}")
    
    assert char_probs_outside.max().item() < 1e-10, "Masking is not effective enough!"
    # 学習モードと評価モードの差（Dropout等）があるため完全一致はしないが、
    # マスキングロジックが共通であれば、爆発的な差は出ないはず
    assert abs(ctc_loss_train.item() - ctc_loss_val.item()) < 10.0, "Massive discrepancy between train and val loss logic!"

if __name__ == "__main__":
    test_train_val_loss_consistency()