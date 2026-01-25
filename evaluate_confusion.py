import torch
import torch.nn as nn
import torchaudio
import numpy as np
import argparse
import os
import csv
from collections import Counter
from typing import List, Tuple

from data_gen import CWDataset, generate_sample
from model import StreamingConformer
import config
from train import levenshtein, map_prosigns, KOCH_CHARS

# Use centralized config
CHARS = config.CHARS
CHAR_TO_ID = config.CHAR_TO_ID
ID_TO_CHAR = config.ID_TO_CHAR
NUM_CLASSES = config.NUM_CLASSES

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = StreamingConformer(
        n_mels=config.N_MELS,
        num_classes=NUM_CLASSES,
        d_model=config.D_MODEL,
        n_head=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Pre-compute Mel filterbank
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=500.0,
        f_max=900.0,
        center=False
    ).to(device)

    # Generate Evaluation Dataset
    # 要件: config.CHARS に含まれる全トークンを均等に（各50回以上）含む
    # 条件: 20 WPM, SNR 100dB (Clean)
    print("Generating evaluation dataset...")
    
    # 全トークン（スペース除く）
    tokens_to_evaluate = [c for c in config.CHARS if c != ' ']
    samples_per_token = args.samples_per_token
    
    confusion_counter = Counter()
    char_stats = {c: {"total": 0, "match": 0, "sub": 0, "del": 0, "subs_to": Counter()} for c in config.CHARS}
    insertion_counter = Counter()
    total_edit_distance = 0
    total_ref_length = 0

    with torch.no_grad():
        for token in tokens_to_evaluate:
            print(f"Evaluating token: {token}")
            for _ in range(samples_per_token):
                # 単一トークンの評価だと文脈が弱すぎる可能性があるため、
                # ランダムな文字を混ぜつつ対象トークンを含める
                length = 5
                other_tokens = torch.utils.data.random_split(tokens_to_evaluate, [1, len(tokens_to_evaluate)-1])[0]
                
                # ターゲットトークンを必ず含める
                test_tokens = [token] + np.random.choice(tokens_to_evaluate, length-1).tolist()
                np.random.shuffle(test_tokens)
                text = "".join(test_tokens)
                
                waveform, ref_text, _ = generate_sample(
                    text, 
                    wpm=args.wpm, 
                    snr_db=args.snr, 
                    jitter=0.0, 
                    weight=1.0
                )
                
                # Inference
                x = waveform.to(device).unsqueeze(0)
                mels = mel_transform(x)
                mels = torch.log1p(mels * 100.0) / 5.0
                mels = mels.transpose(1, 2)
                
                (logits, _), _ = model(mels)
                preds = logits.argmax(dim=2)[0]
                
                # CTC Greedy Decode
                decoded_indices = []
                prev = -1
                for t in range(preds.size(0)):
                    idx = preds[t].item()
                    if idx != 0 and idx != prev:
                        decoded_indices.append(idx)
                    prev = idx
                
                hyp_text = "".join([ID_TO_CHAR.get(idx, "") for idx in decoded_indices]).strip()
                ref_text = ref_text.strip()

                # Calculate Levenshtein and Backtrace
                dist, ops = levenshtein(ref_text, hyp_text)
                total_edit_distance += dist
                # Use mapped length for normalized CER calculation
                total_ref_length += len(map_prosigns(ref_text))

                for op, ref, hyp in ops:
                    if op == 'match':
                        if ref in char_stats:
                            char_stats[ref]["total"] += 1
                            char_stats[ref]["match"] += 1
                    elif op == 'sub':
                        confusion_counter[(ref, hyp)] += 1
                        if ref in char_stats:
                            char_stats[ref]["total"] += 1
                            char_stats[ref]["sub"] += 1
                            char_stats[ref]["subs_to"][hyp] += 1
                    elif op == 'ins':
                        confusion_counter[(None, hyp)] += 1
                        insertion_counter[hyp] += 1
                    elif op == 'del':
                        confusion_counter[(ref, None)] += 1
                        if ref in char_stats:
                            char_stats[ref]["total"] += 1
                            char_stats[ref]["del"] += 1

    # 1. Detailed Character Statistics Table
    print("\n" + "="*80)
    print(f"{'Char':<6} | {'Total':<6} | {'Match':<6} | {'Acc %':<8} | {'Del':<5} | {'Main Substitution'}")
    print("-" * 80)
    
    # Sort by accuracy (ascending) to highlight problematic characters
    sorted_chars = []
    for c in tokens_to_evaluate:
        stats = char_stats[c]
        acc = (stats["match"] / stats["total"] * 100) if stats["total"] > 0 else 0
        sorted_chars.append((c, stats, acc))
    
    # Display all characters
    for c, stats, acc in sorted(sorted_chars, key=lambda x: x[2]):
        main_sub = ""
        if stats["subs_to"]:
            top_sub, sub_count = stats["subs_to"].most_common(1)[0]
            main_sub = f"{top_sub} ({sub_count})"
        
        print(f"{c:<6} | {stats['total']:<6} | {stats['match']:<6} | {acc:>6.1f}% | {stats['del']:<5} | {main_sub}")

    # 2. Insertion Errors Ranking
    print("\n" + "="*80)
    print("Top Insertion Errors ([INS]):")
    for hyp, count in insertion_counter.most_common(10):
        print(f"  - '{hyp}': {count} times")

    # 3. Phase-based Summary
    # Determine which characters are "learned" based on checkpoint if possible,
    # but here we use the KOCH_CHARS order to show progress.
    print("\n" + "="*80)
    print("Phase-based Summary (Koch Order):")
    
    # Assume the model is trained up to some point in KOCH_CHARS
    # We'll group by every 5 characters as a proxy for phases if not explicitly known
    learned_chars = []
    if 'curriculum_phase' in checkpoint:
        phase = checkpoint['curriculum_phase']
        num_learned = 0
        if phase == 1: num_learned = 2
        elif phase == 2: num_learned = 4
        else: num_learned = 4 + (phase - 2)
        learned_chars = list(KOCH_CHARS[:num_learned])
        print(f"Detected Curriculum Phase: {phase} (Learned chars: {''.join(learned_chars)})")

    def calc_group_acc(char_list):
        t = sum(char_stats[c]["total"] for c in char_list if c in char_stats)
        m = sum(char_stats[c]["match"] for c in char_list if c in char_stats)
        return (m / t * 100) if t > 0 else 0

    if learned_chars:
        unlearned_chars = [c for c in tokens_to_evaluate if c not in learned_chars]
        print(f"  - Learned Characters Accuracy:   {calc_group_acc(learned_chars):>6.1f}%")
        print(f"  - Unlearned Characters Accuracy: {calc_group_acc(unlearned_chars):>6.1f}%")
    
    # Generic Koch-based breakdown
    print("  Breakdown by Koch groups:")
    for i in range(0, len(KOCH_CHARS), 5):
        group = list(KOCH_CHARS[i:i+5])
        if not group: break
        acc = calc_group_acc(group)
        print(f"    Group {i//5 + 1} ({''.join(group):<5}): {acc:>6.1f}%")

    # 4. Save Confusion Matrix (Keep existing functionality)
    print("\n" + "="*80)
    print("Saving detailed confusion matrix to confusion_matrix.csv...")
    with open("confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ref", "Hyp", "Count"])
        for (ref, hyp), count in confusion_counter.most_common():
            writer.writerow([ref if ref else "[INS]", hyp if hyp else "[DEL]", count])

    avg_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0
    print(f"\nOverall CER: {avg_cer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model confusion matrix")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--wpm", type=int, default=20, help="WPM for evaluation")
    parser.add_argument("--snr", type=float, default=100.0, help="SNR for evaluation")
    parser.add_argument("--samples-per-token", type=int, default=60, help="Number of samples per token for evaluation")
    
    args = parser.parse_args()
    evaluate(args)