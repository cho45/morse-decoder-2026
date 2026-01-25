import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
import numpy as np
from typing import List, Tuple

# Add current directory to path
sys.path.append(os.getcwd())

from data_gen import CWDataset, MORSE_DICT
from model import StreamingConformer
import config
from train import Trainer # Re-use Trainer class for convenience in loading

def align_strings(ref, hyp):
    """
    Simple Needleman-Wunsch alignment for visualization
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
            
    # Backtrack
    i, j = n, m
    aligned_ref = []
    aligned_hyp = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            aligned_ref.append(ref[i-1])
            aligned_hyp.append(hyp[j-1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1: # Substitution
            aligned_ref.append(ref[i-1])
            aligned_hyp.append(hyp[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1: # Deletion (in hyp)
            aligned_ref.append(ref[i-1])
            aligned_hyp.append("_")
            i -= 1
        else: # Insertion (in hyp)
            aligned_ref.append("_")
            aligned_hyp.append(hyp[j-1])
            j -= 1
            
    return "".join(reversed(aligned_ref)), "".join(reversed(aligned_hyp))

def main():
    parser = argparse.ArgumentParser(description="Diagnose Model Errors")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--samples", type=int, default=20)
    
    # Dummy args for Trainer init
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    
    args = parser.parse_args()
    
    # Initialize Trainer to leverage its setup logic
    trainer = Trainer(args)
    
    # Load latest checkpoint
    checkpoints = [f for f in os.listdir(args.save_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    def get_epoch(filename):
        try: return int(filename.split('_')[-1].split('.')[0])
        except: return 0
    checkpoints.sort(key=get_epoch)
    checkpoint_path = os.path.join(args.save_dir, checkpoints[-1])
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    
    # Handle state dict mismatch if any (re-using logic from train.py)
    model_dict = trainer.model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    trainer.model.load_state_dict(model_dict)
    
    trainer.model.eval()
    
    # Dataset for diagnosis - Use Phase 3 settings (Realistic) to see hard cases
    # Or match current training phase. Let's assume Phase 1/2 for now since user mentioned fixed WPM.
    dataset = CWDataset(num_samples=args.samples)
    # Force settings to match the "Fixed WPM" context mentioned by user
    dataset.min_wpm = 20
    dataset.max_wpm = 20
    dataset.min_snr = 50.0 # Clean-ish
    dataset.max_snr = 100.0
    dataset.jitter_max = 0.0
    dataset.weight_var = 0.0
    dataset.min_len = 5
    dataset.max_len = 10
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=trainer.collate_fn)
    
    print("\n=== Error Diagnosis Report ===\n")
    
    with torch.no_grad():
        for batch_idx, (waveforms, targets, lengths, target_lengths, texts, wpms, signal_targets) in enumerate(dataloader):
            mels, input_lengths = trainer.compute_mels_and_lengths(waveforms, lengths)
            
            (logits, signal_logits), _ = trainer.model(mels)
            
            # Decode
            preds = logits.argmax(dim=2)
            
            for i in range(len(texts)):
                ref_text = texts[i]
                
                # Decode greedy
                length = input_lengths[i].item()
                pred_indices = preds[i, :length]
                
                decoded_indices = []
                decoded_positions = []
                prev = -1
                for t in range(len(pred_indices)):
                    idx = pred_indices[t].item()
                    if idx != 0 and idx != prev:
                        decoded_indices.append(idx)
                        decoded_positions.append(t)
                    prev = idx
                
                hyp_text = "".join([config.ID_TO_CHAR.get(idx, "") for idx in decoded_indices])
                
                # Check for errors
                ref_clean = ref_text.replace(" ", "")
                if ref_clean != hyp_text:
                    print(f"Sample {batch_idx * args.batch_size + i + 1}:")
                    print(f"  REF: {ref_text}")
                    
                    # Align strings for clear diff
                    a_ref, a_hyp = align_strings(ref_clean, hyp_text)
                    print(f"  CMP: {a_ref}")
                    print(f"       {a_hyp}")
                    
                    # Visualize Signal Detection
                    # Downsample logic matches model output
                    sig_len = signal_logits.size(1)
                    sig_target_seq = signal_targets[i, :sig_len, 0].cpu().numpy()
                    sig_pred_seq = torch.sigmoid(signal_logits[i, :, 0]).cpu().numpy()
                    
                    # Create visualization string (compressed 4x for display)
                    vis_len = 80
                    step = max(1, len(sig_target_seq) // vis_len)
                    
                    t_str = ""
                    p_str = ""
                    for j in range(0, len(sig_target_seq), step):
                        val_t = sig_target_seq[j:j+step].max()
                        val_p = sig_pred_seq[j:j+step].max()
                        t_str += '#' if val_t > 0.5 else '.'
                        p_str += '#' if val_p > 0.5 else '.'
                        
                    print(f"  SIG(T): {t_str}")
                    print(f"  SIG(P): {p_str}")
                    
                    # Analyze specific morse patterns if needed
                    # e.g. Did it confuse K (-.-) with C (-.-.)?
                    
                    print("-" * 50)

if __name__ == "__main__":
    main()