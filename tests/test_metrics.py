import pytest
from inference_utils import levenshtein_prosign as levenshtein

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
    # プロサイン正規化により、"<SOS>" は 1文字として扱われる
    ref = "<SOS>"
    hyp = "<SOS>"
    dist, _ = levenshtein(ref, hyp)
    assert dist == 0
    
    # もし hyp が "SN" だった場合、どれくらいの距離になるか
    # "<SOS>" (1トークン) が "S", "O", "S" に置換・挿入されたとみなされる
    # 正規化後: "\x01" vs "SOS"
    # 置換 (\x01->S) + 挿入 (N) = 距離 2
    hyp2 = "SN"
    dist, _ = levenshtein(ref, hyp2)
    assert dist == 2

def test_levenshtein_ops():
    # バックトレース (ops) の正確性を検証
    # match, sub, del, ins の各オペレーションが正しく記録されているか確認

    # 1. 単純な置換
    dist, ops = levenshtein("A", "B")
    assert dist == 1
    assert ops == [('sub', 'A', 'B')]

    # 2. 挿入と削除
    dist, ops = levenshtein("A", "AB")
    assert dist == 1
    assert ops == [('match', 'A', 'A'), ('ins', None, 'B')]

    dist, ops = levenshtein("AB", "A")
    assert dist == 1
    assert ops == [('match', 'A', 'A'), ('del', 'B', None)]

    # 3. Prosign を含む複雑なケース
    # "<SOS>" (正規化済み) -> "S"
    # 正規化後: "\x01" -> "S"
    # 結果: sub("\x01", "S") ではなく、元の文字に戻されているか
    dist, ops = levenshtein("<SOS>", "S")
    assert dist == 1
    assert ops == [('sub', '<SOS>', 'S')]

    # 4. 複数の操作
    # "CQ <SOS>" -> "C <VE>"
    # map_prosigns によりスペースは維持されるが、levenshtein_prosign に渡される前に
    # calculate_cer などではスペースが除去される。ここでは直接呼び出すのでスペースあり。
    dist, ops = levenshtein("CQ <SOS>", "C <VE>")
    # C(match), Q(del), space(match), <SOS>(sub <VE>)
    # 距離: 1(Q del) + 1(<SOS> sub <VE>) = 2
    assert dist == 2
    assert ops == [
        ('match', 'C', 'C'),
        ('del', 'Q', None),
        ('match', ' ', ' '),
        ('sub', '<SOS>', '<VE>')
    ]

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

def test_cer_with_prosigns():
    from inference_utils import calculate_cer
    # Prosign を含む場合の CER 計算の正確性を検証
    # "<SOS> K" vs "<SOS> R"
    # スペースは現在語彙に含まれるため、除去されずそのまま計算される
    # "<SOS> K" -> ["<SOS>", " ", "K"] (length 3)
    # "<SOS> R" -> ["<SOS>", " ", "R"]
    # 距離 1, 長さ 3 -> CER 1/3
    assert calculate_cer("<SOS> K", "<SOS> R") == pytest.approx(1/3)

    # "<SK>" vs "SK"
    # 距離 2 (正規化により <SK> は 1文字, "SK" は 2文字), 長さ 1 -> CER 2.0
    assert calculate_cer("<SK>", "SK") == 2.0

    # 複雑なケース
    # "CQ DE <SOS> K" vs "CQ DE <SOS> R"
    # "CQ DE <SOS> K" -> ["C", "Q", " ", "D", "E", " ", "<SOS>", " ", "K"] (length 9)
    # 距離 1, 長さ 9 -> CER 1/9
    assert calculate_cer("CQ DE <SOS> K", "CQ DE <SOS> R") == pytest.approx(1/9)

if __name__ == "__main__":
    # 手動実行用
    test_levenshtein_basic()
    test_levenshtein_prosigns()
    test_levenshtein_ops()
    test_cer_calculation_logic()
    test_cer_with_prosigns()
    print("Metrics tests passed!")