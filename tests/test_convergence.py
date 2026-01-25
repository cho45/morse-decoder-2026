import torch
import torch.nn as nn
import torch.optim as optim
from train import Trainer
from data_gen import generate_sample
import config
import os

class DummyArgs:
    def __init__(self):
        self.samples_per_epoch = 10
        self.lr = 5e-4 # 少し慎重にする
        self.weight_decay = 0.0 # 過学習を妨げない
        self.grad_clip = 5.0
        self.save_dir = "test_checkpoints"
        self.curriculum_phase = 0
        self.resume = None
        self.freeze_encoder = False

def test_single_sample_convergence():
    """
    1つのサンプルに対して完璧に過学習できるかを確認する。
    これができない設計は、数学的に破綻している。
    """
    args = DummyArgs()
    trainer = Trainer(args)
    trainer.model.train()
    
    # "A" (.-) 1つだけの固定サンプル
    waveform, text, _, _ = generate_sample("A", wpm=20, snr_db=50)
    waveforms = waveform.unsqueeze(0).to(trainer.device)
    lengths = torch.tensor([waveform.size(0)]).to(trainer.device)
    
    # 正解ラベルの作成
    # train.py の collate_fn のロジックを模倣
    targets = torch.tensor([config.CHAR_TO_ID['A']], dtype=torch.long).to(trainer.device)
    target_lengths = torch.tensor([1], dtype=torch.long).to(trainer.device)

    optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-3)
    
    print("\nStarting convergence test on a single sample 'A'...")
    initial_loss = None
    final_loss = None
    
    for i in range(300): # 必要最小限のステップ数に短縮
        optimizer.zero_grad()
        mels, input_lengths = trainer.compute_mels_and_lengths(waveforms, lengths)
        (logits, _, _), _ = trainer.model(mels)
        
        if i == 0:
            print(f"\n[DEBUG] Logits T: {logits.size(1)}, input_lengths: {input_lengths.item()}")

        logits_t = logits.transpose(0, 1).log_softmax(2)
        loss = trainer.criterion(logits_t, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        if initial_loss is None: initial_loss = loss.item()
        if i % 50 == 0:
            print(f"Step {i} | Loss: {loss.item():.6f}")
        final_loss = loss.item()

    print(f"Final Loss after 300 steps: {final_loss:.6f}")
    
    # 物理的に正しい設計なら、1つのサンプルに対する Loss は確実に減少するはず。
    # 300ステップでは 0.15 以下を目標とする（マルチタスク化による収束速度の変化を考慮）。
    assert final_loss < 0.15, f"Model failed to converge on a single sample! Final Loss: {final_loss}"

if __name__ == "__main__":
    test_single_sample_convergence()