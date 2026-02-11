import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import string
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import StreamingConformer
from data_gen import generate_sample, CWDataset, MorseGenerator
from inference_utils import preprocess_waveform, decode_multi_task, calculate_cer
import config

from diagnostics.snr_eval_utils import SyntheticMorseDataset

class PerformanceEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint from {checkpoint_path} on {self.device}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model = StreamingConformer(
            n_mels=config.N_BINS,
            num_classes=config.NUM_CLASSES,
            d_model=config.D_MODEL,
            n_head=config.N_HEAD,
            num_layers=config.NUM_LAYERS,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict[float, List[float]]:
        # Map SNR -> List of CERs
        results = defaultdict(list)
        
        with torch.no_grad():
            for waveforms, actual_texts, wpms, freqs, snrs in tqdm(dataloader, desc="Evaluating"):
                waveforms = waveforms.to(self.device)
                mels = preprocess_waveform(waveforms, self.device)
                
                states = self.model.get_initial_states(mels.size(0), self.device)
                (ctc_batch, sig_batch, bound_batch), _ = self.model(mels, states)
                
                # Move to CPU for decoding
                ctc_batch = ctc_batch.cpu()
                sig_batch = sig_batch.cpu()
                bound_probs_batch = torch.sigmoid(bound_batch).squeeze(-1).cpu()

                for i, text in enumerate(actual_texts):
                    decoded, _ = decode_multi_task(ctc_batch[i], sig_batch[i], bound_probs_batch[i])
                    cer = calculate_cer(text, decoded)
                    
                    snr_val = float(snrs[i].item())
                    results[snr_val].append(cer)
                    
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples", type=int, default=50, help="Samples per SNR point")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_snr_performance.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization")
    parser.add_argument("--fading-speed", type=float, default=0.0)
    parser.add_argument("--min-fading", type=float, default=1.0)
    parser.add_argument("--qrm-prob", type=float, default=0.0)
    parser.add_argument("--impulse-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of data loading workers")
    args = parser.parse_args()

    evaluator = PerformanceEvaluator(args.checkpoint)
    dataset_source = CWDataset() # For phrase generation
    snrs = np.arange(config.EVAL_SNR_MIN, config.EVAL_SNR_MAX, config.EVAL_SNR_STEP)
    
    print(f"Starting evaluation with {args.samples} samples per SNR point...")
    print(f"Settings: Fading={args.fading_speed}, MinFading={args.min_fading}, QRM={args.qrm_prob}, Impulse={args.impulse_prob}")
    print(f"Parallel data loading with {args.workers} workers, batch size {args.batch_size}")
    print(f"Target SNRs: {snrs}")

    # 1. Random High Density
    print("\nRunning Random High Density Evaluation...")
    random_dataset = SyntheticMorseDataset(
        samples_per_snr=args.samples, snrs=snrs, dataset=dataset_source, wpm=15,
        random_freq=args.random_freq, type='random',
        fading_speed=args.fading_speed, min_fading=args.min_fading,
        qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
    )
    random_loader = DataLoader(
        random_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, shuffle=False, drop_last=False
    )
    
    random_results = evaluator.evaluate_dataloader(random_loader)
    
    # Calculate means properly sorted by SNR
    random_avg_cers = [np.mean(random_results[snr]) for snr in snrs]
    
    # 2. Packed Phrases
    print("\nRunning Packed Phrases Evaluation...")
    phrase_dataset = SyntheticMorseDataset(
        samples_per_snr=args.samples, snrs=snrs, dataset=dataset_source, wpm=15,
        random_freq=args.random_freq, type='phrase',
        fading_speed=args.fading_speed, min_fading=args.min_fading,
        qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
    )
    phrase_loader = DataLoader(
        phrase_dataset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, drop_last=False
    )
    
    phrase_results = evaluator.evaluate_dataloader(phrase_loader)
    phrase_avg_cers = [np.mean(phrase_results[snr]) for snr in snrs]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(snrs, random_avg_cers, marker='o', label='Random 6-char (Avg CER)')
    plt.plot(snrs, phrase_avg_cers, marker='s', label='Standard Phrases (Avg CER)')
    
    plt.axhline(y=0.5, color='yellow', linestyle='-.', alpha=0.5, label='CER 50% (Physical Limit)')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='CER 10% (Usable)')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='CER 5% (Near Perfect)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (in 2500Hz BW) [dB]")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"Model Robustness: SNR_2500 vs CER\nCheckpoint: {os.path.basename(args.checkpoint)}")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis() # Better is up
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()