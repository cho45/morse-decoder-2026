"""
MorseEncoder クラスのユニットテスト
"""
import unittest
from data_gen import MorseEncoder
import config


class TestMorseEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = MorseEncoder()
    
    def test_text_to_morse_code_basic(self):
        """基本文字のモールスコード変換"""
        self.assertEqual(self.encoder.text_to_morse_code("A"), ".-")
        self.assertEqual(self.encoder.text_to_morse_code("B"), "-...")
        self.assertEqual(self.encoder.text_to_morse_code("C"), "-.-.")
        self.assertEqual(self.encoder.text_to_morse_code("HELLO"), ".... . .-.. .-.. ---")
    
    def test_text_to_morse_code_with_prosigns(self):
        """Prosignsの正しい処理"""
        self.assertEqual(self.encoder.text_to_morse_code("<SK>"), "...-.-")
        self.assertEqual(self.encoder.text_to_morse_code("<KA>"), "-.-.-")
        self.assertEqual(self.encoder.text_to_morse_code("<VE>"), "...-.")
        self.assertEqual(self.encoder.text_to_morse_code("<SOS>"), "...---...")
    
    def test_text_to_morse_code_with_spaces(self):
        """スペースを含むテキストの処理"""
        self.assertEqual(self.encoder.text_to_morse_code("A B"), ".- -...")
        self.assertEqual(self.encoder.text_to_morse_code("CQ CQ"), "-.-. --.- -.-. --.-")
    
    def test_text_to_morse_code_empty(self):
        """空文字列の処理"""
        self.assertEqual(self.encoder.text_to_morse_code(""), "")
    
    def test_text_to_tokens_basic(self):
        """基本文字のトークン化"""
        tokens = self.encoder.text_to_tokens("ABC")
        self.assertEqual(tokens, ["A", "B", "C"])
    
    def test_text_to_tokens_with_prosigns(self):
        """Prosignsの正しいトークン化"""
        tokens = self.encoder.text_to_tokens("CQ<SK>")
        self.assertEqual(tokens, ["C", "Q", "<SK>"])
        
        tokens = self.encoder.text_to_tokens("<KA>DE")
        self.assertEqual(tokens, ["<KA>", "D", "E"])
    
    def test_text_to_tokens_with_spaces(self):
        """スペースを含むテキストのトークン化"""
        tokens = self.encoder.text_to_tokens("A B C")
        self.assertEqual(tokens, ["A", " ", "B", " ", "C"])
    
    def test_text_to_tokens_empty(self):
        """空文字列のトークン化"""
        tokens = self.encoder.text_to_tokens("")
        self.assertEqual(tokens, [])
    
    def test_text_to_tokens_with_multi_char_prosigns(self):
        """複数文字のProsignsの処理（config.PROSIGNSに基づく）"""
        # CQ, DE などは config.PROSIGNS に含まれている場合、単一トークンとして扱われる
        text = "CQ DE"
        tokens = self.encoder.text_to_tokens(text)
        # PROSIGNSに含まれるか確認
        if "CQ" in config.PROSIGNS:
            self.assertIn("CQ", tokens)
        if "DE" in config.PROSIGNS:
            self.assertIn("DE", tokens)
    
    def test_count_units_in_tokens_basic(self):
        """基本トークンのユニット数計算"""
        tokens = ["A", "B", "C"]
        # A: 5, B: 9, C: 11, Spaces: 3*2=6. Total: 31
        units = self.encoder.count_units_in_tokens(tokens)
        self.assertEqual(units, 31)

    def test_count_units_in_tokens_with_spaces(self):
        """スペースを含むトークンのユニット数計算"""
        tokens = ["A", " ", "B"]
        # A: 5, Space: 7, B: 9. Total: 21
        units = self.encoder.count_units_in_tokens(tokens)
        self.assertEqual(units, 21)

    def test_count_units_in_tokens_with_prosigns(self):
        """Prosignsのユニット数計算"""
        tokens = ["<SK>", "C", "Q"]
        # <SK>: 15, C: 11, Q: 13, Spaces: 3*2=6. Total: 45
        units = self.encoder.count_units_in_tokens(tokens)
        self.assertEqual(units, 45)

    def test_count_units_in_tokens_with_weight(self):
        """weightパラメータのユニット数への影響"""
        tokens = ["A", "B"]
        # A (.-): 1*0.5+1+3*0.5 = 3 units
        # B (-...): 3*0.5+1+1*0.5+1+1*0.5+1+1*0.5 = 6 units
        # Space between A-B: 3 units. Total: 12
        units = self.encoder.count_units_in_tokens(tokens, weight=0.5)
        self.assertEqual(units, 12)
    
    def test_count_units_in_tokens_empty(self):
        """空トークンリストのユニット数計算"""
        units = self.encoder.count_units_in_tokens([])
        self.assertEqual(units, 0)
    
    def test_count_units_in_tokens_unknown_token(self):
        """未知のトークンの処理"""
        tokens = ["A", "#", "B"]  # #は未知
        # A: 5, Space A-#: 3, #: 0, Space #-B: 3, B: 9. Total: 20
        units = self.encoder.count_units_in_tokens(tokens)
        self.assertEqual(units, 20)


if __name__ == '__main__':
    unittest.main()
