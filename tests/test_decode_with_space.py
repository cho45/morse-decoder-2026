import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from inference_utils import decode_multi_task

def test_decode_ctc_space():
    """CTC がスペースを出力した場合に正しくデコードされることを確認"""
    # CHAR_TO_ID を使って ID を取得
    space_id = config.CHAR_TO_ID[' ']
    c_id = config.CHAR_TO_ID['C']
    q_id = config.CHAR_TO_ID['Q']
    
    # CTC Logits: C, Q, Space, C, Q, Space
    # (T, NUM_CLASSES)
    T = 20
    logits = torch.full((T, config.NUM_CLASSES), -10.0)
    # Greedy decoding なので、スパイクを立てる
    logits[2, c_id] = 10.0
    logits[5, q_id] = 10.0
    logits[8, space_id] = 10.0
    logits[11, c_id] = 10.0
    logits[14, q_id] = 10.0
    logits[17, space_id] = 10.0
    
    # 他の Head はダミー（干渉しないように設定）
    sig_logits = torch.zeros((T, 4))
    bound_probs = torch.zeros((T, 1))
    
    decoded, _ = decode_multi_task(logits, sig_logits, bound_probs)
    
    # 期待値: "CQ CQ "
    # 現在の実装では末尾の strip() により "CQ CQ" になる可能性が高い
    # また、CTC のスペースを処理していないので "CQ  CQ" (Signal Head の介入がなければ) 等になる可能性がある
    assert decoded == "CQ CQ "

def test_decode_with_signal_head_space():
    """Signal Head によるスペース挿入と CTC スペースが共存・整合することを確認"""
    space_id = config.CHAR_TO_ID[' ']
    c_id = config.CHAR_TO_ID['C']
    
    T = 10
    logits = torch.full((T, config.NUM_CLASSES), -10.0)
    logits[2, c_id] = 10.0
    logits[8, space_id] = 10.0
    
    # Signal Head が Word Space (3) を検知
    sig_logits = torch.zeros((T, 4))
    sig_logits[5, 3] = 10.0 # C と Space の間
    
    # Boundary Head も有効
    bound_probs = torch.zeros((T, 1))
    bound_probs[5, 0] = 0.5
    
    decoded, _ = decode_multi_task(logits, sig_logits, bound_probs)
    
    # 重複してスペースが入らないこと、かつ末尾が残ること
    assert decoded == "C "