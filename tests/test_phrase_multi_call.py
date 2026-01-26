import sys
import os
import unittest
from unittest.mock import patch

sys.path.append(os.getcwd())

import config
from data_gen import CWDataset

class TestPhraseMultiCall(unittest.TestCase):
    def setUp(self):
        self.dataset = CWDataset(num_samples=10, phrase_prob=1.0)

    def test_generate_phrase_with_multi_call(self):
        # Mock PHRASE_TEMPLATES to include multi-call placeholders
        test_templates = ["{call1} DE {call2} K"]
        with patch('config.PHRASE_TEMPLATES', test_templates):
            # Generate multiple phrases and check if call1 and call2 are different
            # We try multiple times because random generation might occasionally produce the same callsign
            diff_found = False
            for _ in range(10):
                phrase = self.dataset.generate_phrase()
                parts = phrase.split(" DE ")
                call1 = parts[0]
                call2 = parts[1].split(" ")[0]
                
                print(f"Generated phrase: {phrase} (call1: {call1}, call2: {call2})")
                
                if call1 != call2:
                    diff_found = True
                    break
            
            self.assertTrue(diff_found, "Should generate different callsigns for {call1} and {call2}")

    def test_generate_phrase_backward_compatibility(self):
        # Mock PHRASE_TEMPLATES to include old {call} placeholder
        test_templates = ["DE {call} {call}"]
        with patch('config.PHRASE_TEMPLATES', test_templates):
            phrase = self.dataset.generate_phrase()
            parts = phrase.split(" ")
            # "DE", "CALL", "CALL"
            self.assertEqual(parts[1], parts[2], f"Both {{call}} should be same: {phrase}")

if __name__ == "__main__":
    unittest.main()