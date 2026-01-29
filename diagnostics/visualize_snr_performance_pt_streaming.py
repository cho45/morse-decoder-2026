import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import string
from tqdm import tqdm
from typing import List, Tuple, Dict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import StreamingConformer
from data_gen import generate_sample, CWDataset, MorseGenerator
import config
from inference_utils import preprocess_waveform, decode_multi_task, calculate_cer
from diagnostics.visualize_snr_performance import generate_random_text

class PyTorchStreamingEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading checkpoint from {checkpoint_path} on {self.device}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
        
        states = None # Initial states
        
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

    def evaluate_batch(self, texts: List[str], snr_db: float, wpm: int = 20, random_freq: bool = False) -> List[float]:
        cers = []
        for text in texts:
            freq = random.uniform(config.MIN_FREQ, config.MAX_FREQ) if random_freq else 700.0
            
            # Adaptive WPM for phrases to fit in 10s
            sample_wpm = wpm
            if wpm == 20:
                sample_wpm = self.gen.estimate_wpm_for_target_frames(
                    text,
                    target_frames=int(10.0 * 0.9 * config.SAMPLE_RATE / config.HOP_LENGTH),
                    min_wpm=15, max_wpm=45
                )

            waveform, _, _, _ = generate_sample(
                text=text, wpm=sample_wpm, snr_db=snr_db, frequency=freq,
                jitter=0.0, weight=1.0, fading_speed=0.0, min_fading=1.0
            )
            
            mels = preprocess_waveform(waveform, self.device)
            logits, signal_logits, boundary_logits = self.run_inference_streaming(mels)
            
            bound_probs = torch.sigmoid(boundary_logits[0]).squeeze(-1)
            decoded, _ = decode_multi_task(logits[0], signal_logits[0], bound_probs)
            cer = calculate_cer(text, decoded)
            cers.append(cer)
        return cers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples", type=int, default=30, help="Samples per SNR point")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_snr_performance_pt_streaming.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization")
    args = parser.parse_args()

    evaluator = PyTorchStreamingEvaluator(args.checkpoint)
    snrs = np.arange(-18, -2, 1)
    
    avg_cers = []
    print(f"Evaluating PyTorch Streaming Performance")
    
    for snr in tqdm(snrs):
        texts = [generate_random_text(6) for _ in range(args.samples)]
        cers = evaluator.evaluate_batch(texts, snr, random_freq=args.random_freq)
        avg_cer = np.mean(cers)
        avg_cers.append(avg_cer)
        print(f"  SNR: {snr:3d}dB | Avg CER: {avg_cer:.4f}")

    plt.figure(figsize=(12, 8))
    plt.plot(snrs, avg_cers, marker='o', label='PyTorch Streaming (Avg CER)')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='CER 10%')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.3, label='CER 5%')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"PyTorch Streaming Model Performance: SNR vs CER")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis()
    plt.ylabel("Character Error Rate (CER) - Top is better")
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()