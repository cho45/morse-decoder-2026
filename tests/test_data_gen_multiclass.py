import torch
import numpy as np
import config
from data_gen import CWDataset, generate_sample

def test_multiclass_labels():
    text = "K" # -.-
    wpm = 20
    waveform, label_text, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_db=100)
    
    # 0: Background, 1: Dit, 2: Dah, 3: Intra-char space, 4: Inter-char space, 5: Inter-word space
    unique_labels = torch.unique(signal_labels).tolist()
    print(f"Unique labels in 'K': {unique_labels}")
    
    # K は Dah(2), Intra(3), Dit(1), Intra(3), Dah(2) を含むはず
    assert 1 in unique_labels # Dit
    assert 2 in unique_labels # Dah
    assert 3 in unique_labels # Intra-char space
    # 0 (Background) は最初と最後にあるはず
    assert 0 in unique_labels

def test_multiclass_word_space():
    text = "K M"
    wpm = 20
    waveform, label_text, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_db=100)
    
    unique_labels = torch.unique(signal_labels).tolist()
    print(f"Unique labels in 'K M': {unique_labels}")
    
    assert 5 in unique_labels # Inter-word space

def test_dataset_multiclass():
    dataset = CWDataset(num_samples=10, min_len=2, max_len=5)
    waveform, label, wpm, signal_labels, boundary_labels = dataset[0]
    
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