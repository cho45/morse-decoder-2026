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
        self.batch_size = 1
        self.grad_clip = 5.0

def test_illegal_spike_penalty():
    """
    境界外で文字が発火した際に、Trainer.compute_loss がそれを正しく罰することを検証する。
    ロジックをコピーせず、Trainer クラスをブラックボックスとしてテストする。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = MockArgs()
    trainer = Trainer(args)
    trainer.model.to(device)
    trainer.model.train()
    
    seq_len = 50
    mels = torch.randn(1, seq_len, config.N_BINS).to(device)
    
    # 全て Blank のターゲット
    targets = torch.tensor([1], dtype=torch.long).to(device)
    target_lengths = torch.tensor([1], dtype=torch.long).to(device)
    input_lengths = torch.tensor([seq_len // 2], dtype=torch.long).to(device)
    
    # 境界フラグを一切立てない（＝どこで発火しても違法）
    boundary_targets = torch.zeros(1, seq_len // 2, dtype=torch.float32).to(device)
    signal_targets = torch.zeros(1, seq_len // 2, dtype=torch.long).to(device)

    # 1. 正常な（Blankに近い）ロジットでの Loss
    with torch.no_grad():
        (logits_clean, signal_logits, boundary_logits), _ = trainer.model(mels)
        # Blank を優勢にする
        logits_clean[:, :, 0] = 10.0
        logits_clean[:, :, 1:] = -10.0
        
        total_loss_clean, loss_dict_clean = trainer.compute_loss(
            logits_clean, signal_logits, boundary_logits,
            targets, target_lengths, input_lengths,
            signal_targets, boundary_targets
        )

    # 2. 境界外で発火（MやKを出す）したロジットでの Loss
    logits_dirty = logits_clean.clone()
    logits_dirty[:, :, 1:] = 20.0 # 境界外で激しく発火
    
    total_loss_dirty, loss_dict_dirty = trainer.compute_loss(
        logits_dirty, signal_logits, boundary_logits,
        targets, target_lengths, input_lengths,
        signal_targets, boundary_targets
    )

    print(f"Total Loss (Clean): {total_loss_clean.item():.6f} (Penalty: {loss_dict_clean['penalty'].item():.6f})")
    print(f"Total Loss (Dirty): {total_loss_dirty.item():.6f} (Penalty: {loss_dict_dirty['penalty'].item():.6f})")

    # 境界外での発火が、物理法則（Penalty Loss）によって厳罰に処されていることを確認
    # Dirty な状態の方が、Clean な状態よりも明らかに Loss が高くなければならない
    assert total_loss_dirty.item() > total_loss_clean.item() + 10.0, "System is ignoring illegal spikes!"
    # 確率の和（Penalty）は最大でも 1.0 程度（1フレームあたり）になるため、閾値を 0.5 に調整
    assert loss_dict_dirty['penalty'].item() > loss_dict_clean['penalty'].item() + 0.5, "Penalty loss did not increase!"

if __name__ == "__main__":
    test_illegal_spike_penalty()