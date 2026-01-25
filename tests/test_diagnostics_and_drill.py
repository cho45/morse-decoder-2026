import unittest
import torch
from train import map_prosigns, unmap_prosigns, levenshtein, Trainer
import config

class TestDiagnosticsAndDrill(unittest.TestCase):
    def test_prosign_mapping(self):
        # プロサインが正しく1文字の制御コードに変換・復元されるか
        original = "CQ DE <BT> K"
        mapped = map_prosigns(original)
        # <BT> のみが1文字に置換されているはず (CQ, DE は置換されない)
        # 元の長さ: 2+1+2+1+4+1+1 = 12
        # マップ後: 2+1+2+1+1+1+1 = 9
        self.assertEqual(len(mapped), 9)
        
        unmapped = unmap_prosigns(mapped)
        self.assertEqual(unmapped, original)

    def test_levenshtein_with_prosigns(self):
        # プロサインを1トークンとして編集距離が計算されるか
        ref = "K <BT>"
        hyp = "K <AR>" # <BT> が <AR> に置換されたケース
        
        dist, ops = levenshtein(ref, hyp)
        
        # プロサイン正規化により、置換1回（距離1）と判定されるべき
        self.assertEqual(dist, 1)
        
        # 置換操作の内容を確認
        subs = [op for op in ops if op[0] == 'sub']
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0][1], "<BT>")
        self.assertEqual(subs[0][2], "<AR>")

    def test_drill_mode_logic(self):
        # Trainer のドリルモード発動ロジックの検証
        class MockArgs:
            samples_per_epoch = 1000
            lr = 3e-4
            freeze_encoder = False
            curriculum_phase = 9
            batch_size = 16
            save_dir = "checkpoints_test"
            grad_clip = 5.0

        trainer = Trainer(MockArgs())
        trainer.current_phase = 9
        
        # ケース1: 停滞していない場合 (phases_since_last_advance <= 5)
        trainer.phases_since_last_advance = 3
        trainer.train_epoch(1) # 内部で focus_prob が設定される
        self.assertEqual(trainer.train_dataset.focus_prob, 0.5)

        # ケース2: 停滞している場合 (phases_since_last_advance > 5)
        # ドリルモードが発動し、focus_prob が 0.8 になるはず
        trainer.phases_since_last_advance = 6
        trainer.train_epoch(2)
        self.assertEqual(trainer.train_dataset.focus_prob, 0.8)
        
        # W が Focus に含まれている場合、CONFUSION_PAIRS の PAGJ も Focus に追加されているはず
        # train_epoch 実行後の train_dataset.focus_chars を検証
        # 注意: CONFUSION_PAIRS に基づき、WORPA が Focus になっていることをログから確認済み
        focus_chars = trainer.train_dataset.focus_chars
        for c in "ORPA": # W 以外の、ドリルモードで追加された文字
            self.assertIn(c, focus_chars)

if __name__ == '__main__':
    unittest.main()