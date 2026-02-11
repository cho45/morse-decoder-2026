import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import StreamingConformer
from data_gen import CWDataset, MorseGenerator
import config
from inference_utils import preprocess_waveform, decode_multi_task, calculate_cer
from diagnostics.snr_eval_utils import SyntheticMorseDataset

class PyTorchStreamingEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
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
        
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0,
            center=False
        ).to(self.device)
        
        self.f_bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
        self.f_bin_end = self.f_bin_start + config.N_BINS
        self.gen = MorseGenerator()

    def run_inference_streaming(self, mels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run streaming inference on full mels by chunking, matching ONNX version logic."""
        batch_size = mels.size(0)
        seq_len = mels.size(1)
        chunk_size = 40 # Same as ONNX evaluator
        
        states = self.model.get_initial_states(batch_size, self.device)
        
        all_logits = []
        all_signal_logits = []
        all_boundary_logits = []
        
        with torch.no_grad():
            for i in range(0, seq_len, chunk_size):
                chunk = mels[:, i:i+chunk_size, :]
                
                # Ensure chunk size is multiple of 4 for subsampling (matching ONNX logic)
                if chunk.size(1) % 4 != 0:
                    pad_len = 4 - (chunk.size(1) % 4)
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_len))
                
                (logits, signal_logits, boundary_logits), states = self.model(chunk, states)
                
                all_logits.append(logits)
                all_signal_logits.append(signal_logits)
                all_boundary_logits.append(boundary_logits)
                
        full_logits = torch.cat(all_logits, dim=1)
        full_signal_logits = torch.cat(all_signal_logits, dim=1)
        full_boundary_logits = torch.cat(all_boundary_logits, dim=1)
        
        # Trim back to original length (subsampled)
        expected_len = (seq_len + config.SUBSAMPLING_RATE - 1) // config.SUBSAMPLING_RATE
        full_logits = full_logits[:, :expected_len, :]
        full_signal_logits = full_signal_logits[:, :expected_len, :]
        full_boundary_logits = full_boundary_logits[:, :expected_len, :]
        
        return full_logits, full_signal_logits, full_boundary_logits

    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict[float, List[float]]:
        # Map SNR -> List of CERs
        results = defaultdict(list)
        
        with torch.no_grad():
            for waveforms, actual_texts, wpms, freqs, snrs in tqdm(dataloader, desc="Evaluating"):
                waveforms = waveforms.to(self.device)
                mels = preprocess_waveform(waveforms, self.device)
                
                # Streaming inference simulation
                logits, signal_logits, boundary_logits = self.run_inference_streaming(mels)

                bound_probs_batch = torch.sigmoid(boundary_logits).squeeze(-1)
                
                # Move to CPU for decoding
                logits = logits.cpu()
                signal_logits = signal_logits.cpu()
                bound_probs_batch = bound_probs_batch.cpu()

                for i, text in enumerate(actual_texts):
                    decoded, _ = decode_multi_task(
                        logits[i],
                        signal_logits[i],
                        bound_probs_batch[i]
                    )
                    cer = calculate_cer(text, decoded)
                    
                    snr_val = float(snrs[i].item())
                    results[snr_val].append(cer)
                    
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples", type=int, default=30, help="Samples per SNR point")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_snr_performance_pt_streaming.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization")
    parser.add_argument("--fading-speed", type=float, default=0.0)
    parser.add_argument("--min-fading", type=float, default=1.0)
    parser.add_argument("--qrm-prob", type=float, default=0.0)
    parser.add_argument("--impulse-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of data loading workers")
    args = parser.parse_args()

    snrs = np.arange(config.EVAL_SNR_MIN, config.EVAL_SNR_MAX, config.EVAL_SNR_STEP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating PyTorch Streaming Performance on {device}")
    
    evaluator = PyTorchStreamingEvaluator(args.checkpoint, device)
    dataset_source = CWDataset()
    
    # 1. Random High Density
    random_dataset = SyntheticMorseDataset(
        samples_per_snr=args.samples // 2, snrs=snrs, dataset=dataset_source, wpm=15,
        random_freq=args.random_freq, type='random',
        fading_speed=args.fading_speed, min_fading=args.min_fading,
        qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
    )
    random_loader = DataLoader(
        random_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, shuffle=False, drop_last=False
    )
    random_results = evaluator.evaluate_dataloader(random_loader)
    
    # 2. Packed Phrases
    phrase_dataset = SyntheticMorseDataset(
        samples_per_snr=args.samples // 2, snrs=snrs, dataset=dataset_source, wpm=15,
        random_freq=args.random_freq, type='phrase',
        fading_speed=args.fading_speed, min_fading=args.min_fading,
        qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
    )
    phrase_loader = DataLoader(
        phrase_dataset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, drop_last=False
    )
    phrase_results = evaluator.evaluate_dataloader(phrase_loader)
    
    # Combine results
    avg_cers = []
    for snr in snrs:
        all_cers = random_results[snr] + phrase_results[snr]
        avg_cers.append(np.mean(all_cers))

    plt.figure(figsize=(12, 8))
    plt.plot(snrs, avg_cers, marker='o', label='PyTorch Streaming (Avg CER)')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='CER 10%')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.3, label='CER 5%')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (in 2500Hz BW) [dB]")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"PyTorch Streaming Model Performance: SNR_2500 vs CER")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis()
    plt.ylabel("Character Error Rate (CER) - Top is better")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()