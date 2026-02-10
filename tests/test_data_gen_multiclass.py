import torch
import numpy as np
import config
from data_gen import CWDataset, generate_sample
from config import SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE

def test_multiclass_labels():
    text = "K" # -.-
    wpm = 20
    waveform, label_text, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_2500=100)
    
    # SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE
    unique_labels = torch.unique(signal_labels).tolist()
    print(f"Unique labels in 'K': {unique_labels}")
    
    # K は Dah, Blank, Dit を含むはず
    assert SIG_ID_DIT in unique_labels # Dit
    assert SIG_ID_DAH in unique_labels # Dah
    assert SIG_ID_BLANK in unique_labels # Space

def test_multiclass_word_space():
    text = "K M"
    wpm = 20
    waveform, label_text, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_2500=100)
    
    unique_labels = torch.unique(signal_labels).tolist()
    print(f"Unique labels in 'K M': {unique_labels}")
    
    assert SIG_ID_WORD_SPACE in unique_labels # Inter-word space

def test_dataset_multiclass():
    dataset = CWDataset(num_samples=10, min_len=2)
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[0]
    
    assert signal_labels.dim() == 1
    assert signal_labels.dtype == torch.float32 # DataLoader で float になるが中身は整数
    
    # クラス範囲チェック
    assert torch.all(signal_labels >= 0)
    assert torch.all(signal_labels < config.NUM_SIGNAL_CLASSES)

if __name__ == "__main__":
    test_multiclass_labels()
    test_multiclass_word_space()
    test_dataset_multiclass()
    print("All multiclass data_gen tests passed!")