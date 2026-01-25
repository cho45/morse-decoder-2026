import pytest
from train import levenshtein

def test_levenshtein_basic():
    # 基本的な編集距離の確認
    # 新しい levenshtein は (distance, ops) のタプルを返す
    dist, _ = levenshtein("KM", "KM")
    assert dist == 0
    dist, _ = levenshtein("KM", "MM")
    assert dist == 1
    dist, _ = levenshtein("K", "")
    assert dist == 1
    dist, _ = levenshtein("", "M")
    assert dist == 1
    dist, _ = levenshtein("KMMK", "KM")
    assert dist == 2

def test_levenshtein_prosigns():
    # 略符号（Prosigns）を含む場合の挙動確認
    # プロサイン正規化により、"<BT>" は 1文字として扱われる
    ref = "<BT>"
    hyp = "<BT>"
    dist, _ = levenshtein(ref, hyp)
    assert dist == 0
    
    # もし hyp が "BT" だった場合、どれくらいの距離になるか
    # "<BT>" (1トークン) が "B", "T" (2トークン) に置換・挿入されたとみなされる
    # 正規化後: "\x01" vs "BT"
    # 置換 (\x01->B) + 挿入 (T) = 距離 2
    hyp2 = "BT"
    dist, _ = levenshtein(ref, hyp2)
    assert dist == 2

def test_cer_calculation_logic():
    # train.py 内で行われている CER 計算のシミュレーション
    total_edit_distance = 0
    total_ref_length = 0
    
    samples = [
        ("KM KM", "MM MM"), # 2 errors, length 5 (including space)
        ("K", "M"),         # 1 error, length 1
        ("M", ""),          # 1 error, length 1
    ]
    
    for ref, hyp in samples:
        dist, _ = levenshtein(ref, hyp)
        total_edit_distance += dist
        total_ref_length += len(ref)
        
    avg_cer = total_edit_distance / total_ref_length
    
    # (2 + 1 + 1) / (5 + 1 + 1) = 4 / 7 = 0.5714...
    assert avg_cer == pytest.approx(4/7)

if __name__ == "__main__":
    # 手動実行用
    test_levenshtein_basic()
    test_levenshtein_prosigns()
    test_cer_calculation_logic()
    print("Metrics tests passed!")