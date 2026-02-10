"""
TextGenerator クラスのユニットテスト
"""
import unittest
from data_gen import TextGenerator
import config


class TestTextGenerator(unittest.TestCase):
    def setUp(self):
        self.text_gen = TextGenerator()
    
    def test_generate_random_callsign_basic(self):
        """基本コールサイン生成のテスト"""
        callsign = self.text_gen.generate_random_callsign()
        
        # コールサインの形式: [1-2文字][数字][1-3文字]
        if '/' in callsign:
            # モバイルオペレーション: [1-2文字][数字][1-3文字]/[数字]
            self.assertGreaterEqual(len(callsign), 5)
            self.assertLessEqual(len(callsign), 8)
            # 数字が2つ含まれること
            digit_count = sum(1 for c in callsign if c.isdigit())
            self.assertEqual(digit_count, 2, "Mobile callsign should have exactly two digits")
        else:
            # 通常のコールサイン: [1-2文字][数字][1-3文字]
            self.assertGreaterEqual(len(callsign), 3)
            self.assertLessEqual(len(callsign), 6)
            # 数字が1つ含まれること
            digit_count = sum(1 for c in callsign if c.isdigit())
            self.assertEqual(digit_count, 1, "Callsign should have exactly one digit")
    
    def test_generate_random_callsign_mobile(self):
        """モバイルオペレーションのテスト"""
        # ランダム性のため、複数回試行してモバイルコールサインを検証
        mobile_found = False
        for _ in range(100):
            callsign = self.text_gen.generate_random_callsign()
            if '/' in callsign:
                mobile_found = True
                # モバイルコールサインの形式: /[数字]
                parts = callsign.split('/')
                self.assertEqual(len(parts), 2, "Mobile callsign should have 2 parts")
                self.assertTrue(parts[1].isdigit(), "Mobile suffix should be a digit")
                break
        
        self.assertTrue(mobile_found, "Should find at least one mobile callsign in 100 attempts")
    
    def test_generate_phrase_basic(self):
        """基本フレーズ生成のテスト"""
        phrase = self.text_gen.generate_phrase()
        
        # フレーズが空でないこと
        self.assertGreater(len(phrase), 0)
        
        # フレーズがスペースで終わること
        self.assertTrue(phrase.endswith(" "), "Phrase should end with space")
    
    def test_generate_phrase_with_template(self):
        """テンプレートを使用したフレーズ生成のテスト"""
        template = "{call} {call1} de {call2}"
        callsigns = ("JF1ABC", "JA2DEF")
        
        phrase = self.text_gen.generate_phrase(template=template, callsigns=callsigns)
        
        # 指定したコールサインが含まれること
        self.assertIn("JF1ABC", phrase)
        self.assertIn("JA2DEF", phrase)
    
    def test_generate_random_tokens_basic(self):
        """基本トークン生成のテスト"""
        available_tokens = ["A", "B", "C", "D", "E"]
        min_len = 3
        max_len = 5
        
        tokens = self.text_gen.generate_random_tokens(
            available_tokens, min_len, max_len
        )
        
        # トークン数が範囲内にあること
        self.assertGreaterEqual(len(tokens), min_len)
        self.assertLessEqual(len(tokens), max_len)
        
        # 全てのトークンが使用可能なトークンであること
        for token in tokens:
            self.assertIn(token, available_tokens)
    
    def test_generate_random_tokens_with_spaces(self):
        """スペースが除外されるトークン生成のテスト"""
        available_tokens = ["A", "B", "C", " "]
        min_len = 3
        max_len = 5
        
        tokens = self.text_gen.generate_random_tokens(
            available_tokens, min_len, max_len
        )
        
        # generate_random_tokensはスペースを除外しているため、
        # スペースが含まれないことを確認する
        self.assertNotIn(" ", tokens, "Space should be excluded from random tokens")
        
        # トークン数が範囲内にあること
        self.assertGreaterEqual(len(tokens), min_len)
        self.assertLessEqual(len(tokens), max_len)
    
    def test_generate_random_tokens_with_focus(self):
        """フォーカストークンのテスト"""
        available_tokens = ["A", "B", "C", "D", "E", "F", "G"]
        focus_tokens = ["A", "B"]
        min_len = 5
        max_len = 10
        focus_prob = 1.0
        
        tokens = self.text_gen.generate_random_tokens(
            available_tokens, min_len, max_len, 
            focus_tokens=focus_tokens, focus_prob=focus_prob
        )
        
        # フォーカストークンが含まれること
        has_focus = any(t in focus_tokens for t in tokens)
        self.assertTrue(has_focus, "Should contain focus tokens")
    
    def test_generate_random_tokens_empty_available(self):
        """空の使用可能トークンリストの処理"""
        available_tokens = []
        min_len = 3
        max_len = 5
        
        tokens = self.text_gen.generate_random_tokens(
            available_tokens, min_len, max_len
        )
        
        # 空のトークンリストの場合、空の結果が返されること
        self.assertEqual(len(tokens), 0)
    
    def test_generate_random_tokens_long_gap(self):
        """Long Gapトークンのテスト"""
        available_tokens = ["A", "B", "C", "D", "E"]
        min_len = 3
        max_len = 10
        long_gap_prob = 1.0  # 必ず挿入
        long_gap_duration = 0.5
        
        tokens = self.text_gen.generate_random_tokens(
            available_tokens, min_len, max_len,
            long_gap_prob=long_gap_prob, long_gap_duration=long_gap_duration
        )
        
        # Long Gapトークンが含まれること
        has_gap = any(t.startswith("<GAP:") for t in tokens)
        self.assertTrue(has_gap, "Should contain Long Gap token")
        
        # Long Gapの持続時間が正しいこと
        gap_tokens = [t for t in tokens if t.startswith("<GAP:")]
        if gap_tokens:
            gap_token = gap_tokens[0]
            self.assertIn("0.5", gap_token)
    
    def test_generate_multiple_phrases_basic(self):
        """複数フレーズ生成のテスト"""
        num_phrases = 3
        long_gap_duration = 0.5
        
        phrases = self.text_gen.generate_multiple_phrases(
            num_phrases, long_gap_duration=long_gap_duration
        )
        
        # フレーズ数とGAPトークン数が正しいこと（フレーズ間にGAPが挿入される）
        # 3フレーズ + 2GAP = 5要素
        expected_length = num_phrases + (num_phrases - 1)
        self.assertEqual(len(phrases), expected_length)
        
        # 全ての要素が文字列であること
        for phrase in phrases:
            self.assertIsInstance(phrase, str)
        
        # フレーズがスペースで終わること（GAPトークンは除く）
        for phrase in phrases:
            if not phrase.startswith("<GAP:"):
                self.assertTrue(phrase.endswith(" "), "Phrase should end with space")
    
    def test_generate_multiple_phrases_with_gap(self):
        """フレーズ間のGapトークンのテスト"""
        num_phrases = 2
        long_gap_duration = 1.0
        
        phrases = self.text_gen.generate_multiple_phrases(
            num_phrases, long_gap_duration=long_gap_duration
        )
        
        # Gapトークンが含まれること
        has_gap = any("<GAP:" in p for p in phrases)
        self.assertTrue(has_gap, "Should contain GAP token")
        
        # Gapの持続時間が正しいこと
        gap_phrases = [p for p in phrases if "<GAP:" in p]
        if gap_phrases:
            self.assertIn("1.0", gap_phrases[0])
    
    def test_tokens_to_text_basic(self):
        """トークンからテキストへの結合テスト"""
        tokens = ["A", " ", "B", " ", "C"]
        text = self.text_gen.tokens_to_text(tokens)
        
        self.assertEqual(text, "A B C ")
    
    def test_tokens_to_text_empty(self):
        """空トークンリストの処理"""
        tokens = []
        text = self.text_gen.tokens_to_text(tokens)
        
        self.assertEqual(text, "")
    
    def test_tokens_to_text_with_prosigns(self):
        """Prosignsを含むトークンの結合テスト"""
        tokens = ["<SK>", " ", "C", " ", "Q"]
        text = self.text_gen.tokens_to_text(tokens)
        
        self.assertEqual(text, "<SK> C Q ")


if __name__ == '__main__':
    unittest.main()
