import unittest
import torch
import torch.nn as nn
from train import Trainer
from model import StreamingConformer
import config

class TestTrainBoundary(unittest.TestCase):
    def test_boundary_loss_integration(self):
        # Trainer が境界損失を計算し、勾配が流れるか検証
        class MockArgs:
            samples_per_epoch = 100
            lr = 3e-4
            freeze_encoder = False
            curriculum_phase = 1
            batch_size = 4
            save_dir = "checkpoints_test"
            grad_clip = 5.0

        trainer = Trainer(MockArgs())
        
        # ダミーデータの作成
        batch_size = 4
        time_len = 50
        num_mels = config.N_BINS
        
        mels = torch.randn(batch_size, time_len, num_mels).to(trainer.device)
        input_lengths = torch.full((batch_size,), time_len, dtype=torch.long).to(trainer.device)
        targets = torch.tensor([1, 2, 1, 2], dtype=torch.long).to(trainer.device)
        target_lengths = torch.ones(batch_size, dtype=torch.long).to(trainer.device)
        
        # 信号ラベルと境界ラベル
        signal_targets = torch.zeros(batch_size, time_len, dtype=torch.long).to(trainer.device)
        boundary_targets = torch.zeros(batch_size, time_len, dtype=torch.float32).to(trainer.device)
        # 1フレームだけ境界を立てる
        boundary_targets[:, 10] = 1.0
        
        # Forward & Loss
        trainer.optimizer.zero_grad()
        states = trainer.model.get_initial_states(batch_size, trainer.device)
        (logits, signal_logits, boundary_logits), _ = trainer.model(mels, states)
        
        # マスキングロジックの再現
        logits_masked = logits.clone()
        is_not_boundary = (boundary_targets < 0.5)
        # 境界フラグが 0 のフレームでは文字トークン（ID 1〜）を抑制
        # logits: (B, T, C), is_not_boundary: (B, T)
        T_out = logits.size(1)
        is_not_boundary = (boundary_targets[:, :T_out] < 0.5)
        
        mask_expanded = is_not_boundary.unsqueeze(-1).expand_as(logits[:, :T_out, 1:])
        logits_masked[:, :T_out, 1:][mask_expanded] -= 50.0
        
        logits_t = logits_masked.transpose(0, 1).log_softmax(2)
        # サブサンプリング後の入力長に合わせる
        T_sub = logits.size(1)
        input_lengths_sub = torch.full((batch_size,), T_sub, dtype=torch.long).to(trainer.device)
        ctc_loss = trainer.criterion(logits_t, targets, input_lengths_sub, target_lengths)
        
        # 境界損失
        # ターゲットをサブサンプリング後のサイズに合わせる
        bound_t_sub = boundary_targets[:, :T_sub].unsqueeze(-1)
        mask_sub = torch.ones(batch_size, T_sub).to(trainer.device)
        raw_bound_loss = trainer.boundary_criterion(boundary_logits, bound_t_sub).squeeze(-1)
        bound_loss = (raw_bound_loss * mask_sub).sum() / (mask_sub.sum() + 1e-6)
        
        loss = 5.0 * ctc_loss + 1.0 * bound_loss
        
        # Backward
        loss.backward()
        
        # 境界ヘッドの重みに勾配が流れているか確認
        self.assertIsNotNone(trainer.model.boundary_head.weight.grad)
        self.assertFalse(torch.all(trainer.model.boundary_head.weight.grad == 0))
        
        print(f"Boundary Loss: {bound_loss.item():.4f}, CTC Loss: {ctc_loss.item():.4f}")

if __name__ == '__main__':
    unittest.main()