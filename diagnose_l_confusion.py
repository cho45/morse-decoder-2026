import torch
import torchaudio
import config
from model import StreamingConformer
from data_gen import CWDataset
from config import ID_TO_CHAR, CHAR_TO_ID
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

def levenshtein_detailed(ref, hyp):
    # Standard Levenshtein but returns operations
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d_del = dp[i-1][j] + 1
            d_ins = dp[i][j-1] + 1
            d_sub = dp[i-1][j-1] + cost
            
            dp[i][j] = min(d_del, d_ins, d_sub)
            
            if dp[i][j] == d_sub:
                ops[i][j] = 'match' if cost == 0 else 'sub'
            elif dp[i][j] == d_del:
                ops[i][j] = 'del'
            else:
                ops[i][j] = 'ins'
                
    # Backtrack
    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ops[i][j] in ['match', 'sub']:
            alignment.append((ops[i][j], ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ops[i][j] == 'del'):
            alignment.append(('del', ref[i-1], '-'))
            i -= 1
        else:
            # j > 0 must be true here
            alignment.append(('ins', '-', hyp[j-1]))
            j -= 1
            
    return alignment[::-1]

def diagnose(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model
    model = StreamingConformer(
        n_mels=config.N_MELS,
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        n_head=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # Prepare Dataset focusing on L and R
    # Phase 4 chars: K, M, R, S, T, L
    chars = "KMRSTL"
    dataset = CWDataset(num_samples=500)
    dataset.min_wpm = 20
    dataset.max_wpm = 20
    dataset.min_snr = 100
    dataset.max_snr = 100
    dataset.chars = chars
    dataset.focus_chars = "L"
    dataset.focus_prob = 0.8 # High focus on L to test it
    
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    
    mel_transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=500.0,
            f_max=900.0,
            center=False
        ),
        torchaudio.transforms.AmplitudeToDB()
    ).to(device)
    
    print("Running diagnosis...")
    
    char_stats = {}
    
    for batch in dataloader:
        waveform, text, _, _ = batch[0]
        waveform = waveform.unsqueeze(0).to(device)
        
        # Manually compute mel
        mels = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=500.0,
            f_max=900.0,
            center=False
        ).to(device)(waveform)
        mels = torch.log1p(mels * 100.0) / 5.0
        mels = mels.transpose(1, 2)
        
        with torch.no_grad():
            (logits, _), _ = model(mels)
            preds = logits.argmax(dim=2).squeeze(0)
            
            # Greedy decode
            decoded = []
            prev = -1
            for t in range(preds.size(0)):
                idx = preds[t].item()
                if idx != 0 and idx != prev:
                    decoded.append(ID_TO_CHAR[idx])
                prev = idx
            hyp = "".join(decoded)
            
            # Analyze alignment
            alignment = levenshtein_detailed(text, hyp)
            
            for op, r, h in alignment:
                if r not in char_stats:
                    char_stats[r] = {'total': 0, 'correct': 0, 'confusions': {}}
                
                char_stats[r]['total'] += 1
                
                if op == 'match':
                    char_stats[r]['correct'] += 1
                elif op == 'sub':
                    char_stats[r]['confusions'][h] = char_stats[r]['confusions'].get(h, 0) + 1
                elif op == 'del':
                    char_stats[r]['confusions']['<DEL>'] = char_stats[r]['confusions'].get('<DEL>', 0) + 1
                elif op == 'ins':
                    # Insertion is usually attributed to the previous char or null, hard to map to 'r'
                    # In our alignment, 'ins' has r='-'
                    pass

    print("\n--- Character Performance ---")
    for char in sorted(char_stats.keys()):
        if char == '-': continue # Skip insertion markers
        stats = char_stats[char]
        total = stats['total']
        correct = stats['correct']
        acc = correct / total * 100 if total > 0 else 0
        print(f"Char '{char}': Acc {acc:.1f}% ({correct}/{total})")
        if stats['confusions']:
            print(f"  Confusions: {stats['confusions']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        diagnose(sys.argv[1])
    else:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if checkpoints:
            def get_epoch(filename):
                try:
                    return int(filename.split('_')[-1].split('.')[0])
                except:
                    return 0
            checkpoints.sort(key=get_epoch)
            latest = checkpoints[-1]
            diagnose(os.path.join('checkpoints', latest))
        else:
            print("No checkpoints found.")
