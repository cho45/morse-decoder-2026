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
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import multiprocessing
from functools import partial

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

def decode_worker(args):
    """Worker function for multiprocessing decoding."""
    logits, signal_logits, b_probs, text = args
    hypothesis, _ = decode_multi_task(logits, signal_logits, b_probs)
    return hypothesis, text

def analyze_char_errors_for_snr(trainer, snr, num_samples, batch_size, device):
    """Run analysis for a specific SNR level."""
    print(f"Generating samples for SNR={snr}dB...")
    
    # Configure dataset for specific SNR
    dataset = CWDataset(
        num_samples=num_samples,
        phrase_prob=0.5,
        min_snr_2500=snr,
        max_snr_2500=snr,
        # Ensure we test a challenging but realistic WPM range
        min_wpm=20,
        max_wpm=30
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=trainer.collate_fn)
    
    all_refs = []
    all_hyps = []
    
    # Store data for multiprocessing
    decode_tasks = []

    print(f"Running inference...")
    with torch.no_grad():
        for waveforms, targets, lengths, target_lengths, texts, wpms, signal_targets, boundary_targets, is_phrases in dataloader:
            mels, input_lengths = trainer.compute_mels_and_lengths(waveforms, lengths)
            (logits, signal_logits, boundary_logits), _ = trainer.model(mels)
            
            # Prepare tasks for parallel decoding
            for i in range(logits.size(0)):
                length = input_lengths[i].item()
                # Move tensors to CPU for decoding worker
                l = logits[i, :length].cpu()
                s = signal_logits[i, :length].cpu()
                b = torch.sigmoid(boundary_logits[i, :length]).squeeze(-1).cpu()
                t = texts[i]
                decode_tasks.append((l, s, b, t))

    # Parallel Decoding
    print(f"Decoding {len(decode_tasks)} samples with {multiprocessing.cpu_count()} workers...")
    with multiprocessing.Pool() as pool:
        results = pool.map(decode_worker, decode_tasks)

    print("Calculating metrics...")
    for hypothesis, reference in results:
        ref_clean = reference.strip().replace(" ", "")
        hyp_clean = hypothesis.strip().replace(" ", "")
        
        _, ops = levenshtein_prosign(ref_clean, hyp_clean)
        
        for op, ref_char, hyp_char in ops:
            if op == 'match':
                all_refs.append(ref_char)
                all_hyps.append(ref_char)
            elif op == 'sub':
                all_refs.append(ref_char)
                all_hyps.append(hyp_char)
            elif op == 'del':
                all_refs.append(ref_char)
                all_hyps.append('<DEL>')
            elif op == 'ins':
                all_refs.append('<INS>')
                all_hyps.append(hyp_char)
                
    return all_refs, all_hyps

def analyze_char_errors(checkpoint_path, num_samples=500, batch_size=16, output_prefix=None, snr_levels=[6, -6, -12]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_prefix is None:
        script_base = os.path.splitext(os.path.basename(__file__))[0]
        output_prefix = os.path.join(os.path.dirname(__file__), script_base)
    
    # Load model once
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    
    # Data storage for plots
    snr_results = {}
    
    # Run analysis for each SNR
    for snr in snr_levels:
        print(f"\n{'='*20} Analyzing SNR = {snr} dB {'='*20}")
        refs, hyps = analyze_char_errors_for_snr(trainer, snr, num_samples, batch_size, device)
        snr_results[snr] = (refs, hyps)

    # Visualization
    print(f"\nGenerating comparative visualization...")
    
    # Setup Figure
    fig, axes = plt.subplots(2, len(snr_levels), figsize=(6 * len(snr_levels), 16))
    if len(snr_levels) == 1:
        axes = np.expand_dims(axes, axis=1) # Handle single column case

    # Collect all possible labels across all SNRs to ensure consistent matrices
    all_vocab = set()
    for refs, hyps in snr_results.values():
        all_vocab.update(refs)
        all_vocab.update(hyps)
    
    vocab_list = sorted(list(all_vocab))
    if '<DEL>' in vocab_list: vocab_list.remove('<DEL>')
    if '<INS>' in vocab_list: vocab_list.remove('<INS>')
    
    # Put special tokens at the end
    all_labels = vocab_list + ['<DEL>', '<INS>']
    
    # Create plots for each SNR
    for i, snr in enumerate(snr_levels):
        refs, hyps = snr_results[snr]
        
        # Calculate F1 Score
        clean_refs = [r for r, h in zip(refs, hyps) if r != '<INS>' and h != '<DEL>']
        clean_hyps = [h for r, h in zip(refs, hyps) if r != '<INS>' and h != '<DEL>']
        report = classification_report(clean_refs, clean_hyps, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']

        # Confusion Matrix
        # Filter refs/hyps to ensure they are in all_labels (though they should be)
        valid_indices = [k for k, (r, h) in enumerate(zip(refs, hyps)) if r in all_labels and h in all_labels]
        f_refs = [refs[k] for k in valid_indices]
        f_hyps = [hyps[k] for k in valid_indices]
        
        cm = confusion_matrix(f_refs, f_hyps, labels=all_labels)
        
        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        # Errors only
        cm_errors = cm_norm.copy()
        np.fill_diagonal(cm_errors, 0)
        
        # Plot 1: Full CM (Diagonal)
        ax_top = axes[0, i]
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues', cbar=False,
                   xticklabels=all_labels, yticklabels=all_labels, ax=ax_top)
        ax_top.set_title(f"SNR {snr}dB\nAcc: {accuracy:.1%} | F1: {macro_f1:.3f}")
        ax_top.set_xlabel("Predicted")
        ax_top.set_ylabel("Reference")
        
        # Plot 2: Errors Only
        ax_bot = axes[1, i]
        sns.heatmap(cm_errors, annot=False, fmt='.2f', cmap='YlOrRd', cbar=False,
                   xticklabels=all_labels, yticklabels=all_labels, ax=ax_bot)
        ax_bot.set_title(f"Errors Only (SNR {snr}dB)")
        ax_bot.set_xlabel("Predicted")
        ax_bot.set_ylabel("Reference")

    plt.tight_layout()
    plot_path = f"{output_prefix}.png"
    plt.savefig(plot_path)
    print(f"Comparative plot saved to: {plot_path}")
    
    # Save CSV report for the lowest SNR (worst case)
    worst_snr = min(snr_levels)
    w_refs, w_hyps = snr_results[worst_snr]
    clean_w_refs = [r for r, h in zip(w_refs, w_hyps) if r != '<INS>' and h != '<DEL>']
    clean_w_hyps = [h for r, h in zip(w_refs, w_hyps) if r != '<INS>' and h != '<DEL>']
    report = classification_report(clean_w_refs, clean_w_hyps, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    csv_path = f"{output_prefix}.csv"
    df_report.to_csv(csv_path)
    print(f"Worst case ({worst_snr}dB) detailed report saved to: {csv_path}")

if __name__ == "__main__":
    # Enable multiprocessing support for PyTorch
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Analyze character-level errors of a CW decoder model.")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint. If not provided, uses latest.")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to analyze PER SNR level.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--output", type=str, help="Output prefix for plots and reports.")
    parser.add_argument("--snr", type=str, default="6,-6,-12", help="Comma-separated list of SNR levels to evaluate (e.g. '6,-6,-12')")
    
    args = parser.parse_args()
    
    snr_levels = [int(x) for x in args.snr.split(",")]
    
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
            
    analyze_char_errors(
        checkpoint_path,
        num_samples=args.samples,
        batch_size=args.batch_size,
        output_prefix=args.output,
        snr_levels=snr_levels
    )