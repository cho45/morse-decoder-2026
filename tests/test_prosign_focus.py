import sys
import os
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_gen import CWDataset
import config

def test_prosign_focus_logic():
    """
    PROSIGNS (e.g., <NJ>) が focus_chars に指定された際、
    正しく生成されるか、およびトークンとして分離されるかを検証する。
    """
    num_samples = 50
    # Prosigns を含む curriculum set を模倣
    all_chars = "1EAWJ2IU3SV4H56TNDB7MGZ8O90KLPQXYCF/?.,-()'!&:;=+_\"$@<NJ><SN>"
    focus_chars = "<NJ><SN>"
    
    dataset = CWDataset(
        num_samples=num_samples,
        allowed_chars=all_chars,
        min_len=5,
        max_len=10,
        focus_chars=focus_chars,
        focus_prob=1.0 # 常に focus_chars を優先
    )
    
    found_prosigns = Counter()
    
    for i in range(num_samples):
        _, label, _, _, _, _ = dataset[i]
        
        # label (text) からトークンを抽出
        # CWDataset は内部で単語間空白を入れることがあるので注意
        from data_gen import MorseGenerator
        gen = MorseGenerator()
        tokens = gen.text_to_morse_tokens(label)
        
        for t in tokens:
            if t in ["<NJ>", "<SN>"]:
                found_prosigns[t] += 1
                
    print(f"\nFound Prosigns: {found_prosigns}")
    
    # focus_prob=1.0 なので、少なくともいくつかの Prosigns が含まれているはず
    assert found_prosigns["<NJ>"] > 0 or found_prosigns["<SN>"] > 0, \
        f"Prosigns in focus_chars were not generated. Label samples: {label}"

if __name__ == "__main__":
    test_prosign_focus_logic()