"""
モデルの文字レベルの誤認識を詳細に分析するスクリプト。
混同行列（Confusion Matrix）の生成、Precision/Recall/F1-scoreの算出、
モールス符号長ごとの精度分析を行い、画像およびCSVとして出力します。

使い方:
    docker run --rm -v `pwd`:/workspace cw-decoder python3 diagnostics/analyze_char_errors.py [オプション]

オプション:
    --checkpoint PATH : 分析するチェックポイントのパス（未指定時は最新のものを使用）
    --samples N       : 分析に使用するサンプル数（デフォルト: 500）
    --batch_size N    : 推論時のバッチサイズ（デフォルト: 16）
    --output PREFIX   : 出力ファイル名のプレフィックス（デフォルト: analyze_char_errors）

出力:
    PREFIX.png : 混同行列のヒートマップ
    PREFIX.csv : 文字ごとの詳細統計レポート
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_gen import CWDataset, MORSE_DICT
from train import Trainer
from inference_utils import decode_multi_task, levenshtein_prosign
import config

def get_morse_len(char):
    """Compute the length of a Morse code sequence in units."""
    code = MORSE_DICT.get(char, "")
    length = 0
    for c in code:
        if c == '.': length += 1
        elif c == '-': length += 3
    length += (len(code) - 1) if len(code) > 0 else 0 # intra-char spaces
    return length

def analyze_char_errors(checkpoint_path, num_samples=500, batch_size=16, output_prefix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_prefix is None:
        # Default to the same name as the script but in the diagnostics directory
        script_base = os.path.splitext(os.path.basename(__file__))[0]
        output_prefix = os.path.join(os.path.dirname(__file__), script_base)
    
    # Load model
    class TrainerArgs:
        def __init__(self, bs):
            self.save_dir = "checkpoints"
            self.samples_per_epoch = 1000
            self.batch_size = bs
            self.lr = 1e-4
            self.freeze_encoder = False
            self.curriculum_phase = 0
    
    args = TrainerArgs(batch_size)
    trainer = Trainer(args)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    
    # Use validation dataset
    dataset = CWDataset(num_samples=num_samples, phrase_prob=0.5)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=trainer.collate_fn)
    
    all_refs = []
    all_hyps = []
    
    char_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'subs': 0, 'dels': 0, 'ins': 0})
    confusion_pairs = []

    print(f"Analyzing {num_samples} samples...")
    with torch.no_grad():
        for waveforms, targets, lengths, target_lengths, texts, wpms, signal_targets, boundary_targets, is_phrases in dataloader:
            mels, input_lengths = trainer.compute_mels_and_lengths(waveforms, lengths)
            (logits, signal_logits, boundary_logits), _ = trainer.model(mels)
            
            for i in range(logits.size(0)):
                length = input_lengths[i].item()
                b_probs = torch.sigmoid(boundary_logits[i, :length]).squeeze(-1)
                
                hypothesis, _ = decode_multi_task(
                    logits[i, :length],
                    signal_logits[i, :length],
                    b_probs
                )
                
                reference = texts[i].strip()
                # Remove spaces for character-level analysis
                ref_clean = reference.replace(" ", "")
                hyp_clean = hypothesis.replace(" ", "")
                
                _, ops = levenshtein_prosign(ref_clean, hyp_clean)
                
                for op, ref_char, hyp_char in ops:
                    if op == 'match':
                        char_stats[ref_char]['correct'] += 1
                        char_stats[ref_char]['total'] += 1
                        all_refs.append(ref_char)
                        all_hyps.append(ref_char)
                    elif op == 'sub':
                        char_stats[ref_char]['subs'] += 1
                        char_stats[ref_char]['total'] += 1
                        all_refs.append(ref_char)
                        all_hyps.append(hyp_char)
                        confusion_pairs.append((ref_char, hyp_char))
                    elif op == 'del':
                        char_stats[ref_char]['dels'] += 1
                        char_stats[ref_char]['total'] += 1
                        all_refs.append(ref_char)
                        all_hyps.append('<DEL>')
                    elif op == 'ins':
                        char_stats['<INS>']['ins'] += 1
                        all_refs.append('<INS>')
                        all_hyps.append(hyp_char)

    # 1. Console Output: Summary Statistics
    print("\n" + "="*60)
    print(" CHARACTER PERFORMANCE SUMMARY ")
    print("="*60)
    
    vocab = sorted(list(set(all_refs + all_hyps)))
    if '<DEL>' in vocab: vocab.remove('<DEL>')
    if '<INS>' in vocab: vocab.remove('<INS>')
    
    # Filter out special tokens for standard metrics
    clean_refs = [r for r, h in zip(all_refs, all_hyps) if r != '<INS>' and h != '<DEL>']
    clean_hyps = [h for r, h in zip(all_refs, all_hyps) if r != '<INS>' and h != '<DEL>']
    
    report = classification_report(clean_refs, clean_hyps, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    
    print("\n--- Top 10 Worst Characters (by F1-score) ---")
    worst_chars = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    worst_chars = worst_chars.sort_values('f1-score').head(10)
    print(worst_chars[['precision', 'recall', 'f1-score', 'support']])

    print("\n--- Common Confusions (Ref -> Hyp) ---")
    conf_counts = defaultdict(int)
    for p in confusion_pairs: conf_counts[p] += 1
    sorted_conf = sorted(conf_counts.items(), key=lambda x: x[1], reverse=True)
    for (r, h), count in sorted_conf[:10]:
        print(f"  {r} -> {h}: {count} times")

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(all_refs, all_hyps, labels=vocab + ['<DEL>'])
    # Normalize by row (reference)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=vocab + ['<DEL>'], yticklabels=vocab + ['<DEL>'])
    plt.title(f"Confusion Matrix (Normalized) - {os.path.basename(checkpoint_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.tight_layout()
    plot_path = f"{output_prefix}.png"
    plt.savefig(plot_path)
    print(f"\nConfusion matrix saved to: {plot_path}")

    # 3. Morse Length Analysis
    morse_len_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for char in vocab:
        if char in MORSE_DICT:
            mlen = get_morse_len(char)
            stats = char_stats[char]
            morse_len_stats[mlen]['correct'] += stats['correct']
            morse_len_stats[mlen]['total'] += stats['total']
    
    print("\n--- Accuracy by Morse Length (Units) ---")
    lengths = sorted(morse_len_stats.keys())
    accs = []
    for l in lengths:
        s = morse_len_stats[l]
        acc = (s['correct'] / s['total'] * 100) if s['total'] > 0 else 0
        accs.append(acc)
        print(f"  Length {l:2d}: {acc:6.1f}% ({s['correct']}/{s['total']})")

    # Save detailed stats to CSV
    csv_path = f"{output_prefix}.csv"
    df_report.to_csv(csv_path)
    print(f"Detailed report saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze character-level errors of a CW decoder model.")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint. If not provided, uses latest.")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to analyze.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--output", type=str, help="Output prefix for plots and reports.")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        import glob
        checkpoints = sorted(glob.glob("checkpoints/checkpoint_epoch_*.pt"), key=os.path.getmtime)
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            print("No checkpoints found in checkpoints/ directory.")
            sys.exit(1)
            
    analyze_char_errors(checkpoint_path, num_samples=args.samples, batch_size=args.batch_size, output_prefix=args.output)