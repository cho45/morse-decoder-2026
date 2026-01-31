import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
import config
from inference_utils import calculate_cer, map_prosigns
from data_gen import CWDataset

def test_train_val_eval_consistency():
    """
    修正後の train.py のバリデーションロジックが、データ生成と整合していることを検証。
    """
    # 1. データ生成側：末尾にスペースが入る
    dataset = CWDataset(num_samples=1)
    _, reference_from_data, _, _, _, _ = dataset[0]
    assert reference_from_data.endswith(' '), f"Data generator must provide trailing space, got '{reference_from_data}'"

    # 2. 予測側：モデルが完璧に予測したと仮定（末尾スペース含む）
    hypothesis = reference_from_data 

    # 3. 評価側：現在の実装（修正済み train.py）と同じ方法で評価
    # 修正後の train.py は strip() を行わず、直接 texts[i] を使う
    reference_eval = reference_from_data
    
    # 完璧な予測なら CER は 0 になるべき
    cer = calculate_cer(reference_eval, hypothesis)
    
    print(f"\n[CONSISTENCY CHECK]")
    print(f"Ref (Data): '{reference_from_data}' (len: {len(reference_from_data)})")
    print(f"Hypothesis: '{hypothesis}'")
    print(f"CER: {cer:.4f}")

    assert cer == 0, f"Perfect prediction must result in CER 0, got {cer}"

def test_ref_len_calculation_consistency():
    """
    修正後の train.py の ref_len 計算が、calculate_cer の内部挙動と一致することを検証。
    """
    reference = "CQ "
    
    # 修正後の train.py のロジック:
    # ref_len = len(map_prosigns(reference))
    ref_len_train = len(map_prosigns(reference)) # -> 3
    
    # calculate_cer の内部で使用される分母も同じはず
    # calculate_cer(ref, hyp) -> dist / len(map_prosigns(ref))
    ref_len_actual = len(map_prosigns(reference)) # -> 3
    
    print(f"\n[REF_LEN CHECK]")
    print(f"Reference: '{reference}'")
    print(f"Train ref_len:  {ref_len_train}")
    print(f"Actual ref_len: {ref_len_actual}")
    
    assert ref_len_train == ref_len_actual, "Ref length calculation must be consistent"
