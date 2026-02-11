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
            for waveforms, actual_texts, wpms, freqs, snrs in tqdm(dataloader, desc="Evaluating", leave=False):
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
    parser.add_argument("--samples", type=int, default=50, help="Samples per SNR point per WPM")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_snr_performance_wpms.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization")
    parser.add_argument("--fading-speed", type=float, default=0.0)
    parser.add_argument("--min-fading", type=float, default=1.0)
    parser.add_argument("--qrm-prob", type=float, default=0.0)
    parser.add_argument("--impulse-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of data loading workers")
    parser.add_argument("--wpms", type=int, nargs='+', default=[10, 20, 30, 40], help="List of WPMs to test")
    args = parser.parse_args()

    evaluator = PerformanceEvaluator(args.checkpoint)
    dataset_source = CWDataset() # For initialization needed by SyntheticMorseDataset
    snrs = np.arange(config.EVAL_SNR_MIN, config.EVAL_SNR_MAX, config.EVAL_SNR_STEP)
    
    # Store aggregated results for plotting
    all_results = {}

    print(f"Starting evaluation with {args.samples} samples per SNR point per WPM...")
    print(f"Settings: Fading={args.fading_speed}, MinFading={args.min_fading}, QRM={args.qrm_prob}, Impulse={args.impulse_prob}")
    print(f"Target WPMs: {args.wpms}")
    print(f"Parallel data loading with {args.workers} workers, batch size {args.batch_size}")
    
    for wpm in args.wpms:
        print(f"\nRunning Evaluation for {wpm} WPM...")
        dataset = SyntheticMorseDataset(
            samples_per_snr=args.samples, snrs=snrs, dataset=dataset_source, wpm=wpm,
            random_freq=args.random_freq, type='random', # Default to random in this script
            fading_speed=args.fading_speed, min_fading=args.min_fading,
            qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, 
            num_workers=args.workers, shuffle=False, drop_last=False
        )
        
        results = evaluator.evaluate_dataloader(loader)
        
        # Calculate means properly sorted by SNR
        avg_cers = []
        for snr in snrs:
            # Finding the closest key in results or exact match
            vals = results.get(float(snr), [])
            if not vals:
                # Try finding closest key if direct access fails (floating point issues)
                keys = list(results.keys())
                if keys:
                    closest_key = min(keys, key=lambda x: abs(x - snr))
                    if abs(closest_key - snr) < 1e-5:
                        vals = results[closest_key]
                
            avg_cers.append(np.mean(vals) if vals else 0.0)
            
        all_results[wpm] = avg_cers

    # Plotting
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    for i, wpm in enumerate(args.wpms):
        marker = markers[i % len(markers)]
        plt.plot(snrs, all_results[wpm], marker=marker, label=f'{wpm} WPM (Avg CER)')
    
    plt.axhline(y=0.5, color='yellow', linestyle='-.', alpha=0.5, label='CER 50% (Physical Limit)')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='CER 10% (Usable)')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='CER 5% (Near Perfect)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (in 2500Hz BW) [dB]")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"Model Robustness: SNR vs CER (Multiple WPM)\nCheckpoint: {os.path.basename(args.checkpoint)}")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis() # Better is up
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
