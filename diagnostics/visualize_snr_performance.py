import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import string
from tqdm import tqdm
from typing import List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import StreamingConformer
from data_gen import generate_sample, CWDataset, MorseGenerator
from inference_utils import preprocess_waveform, decode_multi_task, calculate_cer
import config

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
        self.gen = MorseGenerator()

    def evaluate_batch(self, texts: List[str], snr_2500: float, wpm: int = 20, random_freq: bool = False,
                       fading_speed: float = 0.0, min_fading: float = 1.0,
                       qrm_prob: float = 0.1, impulse_prob: float = 0.001) -> List[float]:
        waveforms = []
        sample_wpms = []
        freqs = []
        
        # 1. Generate all waveforms in CPU loop
        for text in texts:
            freq = random.uniform(config.MIN_FREQ, config.MAX_FREQ) if random_freq else 700.0
            sample_wpm = wpm
            if wpm == 20:
                sample_wpm = self.gen.estimate_wpm_for_target_frames(
                    text,
                    target_frames=int(10.0 * 0.9 * config.SAMPLE_RATE / config.HOP_LENGTH),
                    min_wpm=15, max_wpm=45
                )
            
            waveform, _, _, _ = generate_sample(
                text=text, wpm=sample_wpm, snr_2500=snr_2500, frequency=freq,
                jitter=0.0, weight=1.0, fading_speed=fading_speed, min_fading=min_fading,
                qrm_prob=qrm_prob, impulse_prob=impulse_prob
            )
            waveforms.append(waveform)
            sample_wpms.append(sample_wpm)
            freqs.append(freq)
            
        # 2. Batch Preprocessing and Inference on GPU
        waveforms_batch = torch.stack(waveforms).to(self.device)
        mels = preprocess_waveform(waveforms_batch, self.device)
        
        with torch.no_grad():
            states = self.model.get_initial_states(mels.size(0), self.device)
            (ctc_batch, sig_batch, bound_batch), _ = self.model(mels, states)
            # Move results back to CPU for decoding
            ctc_batch = ctc_batch.cpu()
            sig_batch = sig_batch.cpu()
            bound_probs_batch = torch.sigmoid(bound_batch).squeeze(-1).cpu()
            
        # 3. Decoding and CER Calculation in CPU loop
        cers = []
        for i, text in enumerate(texts):
            decoded, _ = decode_multi_task(ctc_batch[i], sig_batch[i], bound_probs_batch[i])
            cer = calculate_cer(text, decoded)
            cers.append(cer)
            
            # Debug display for first few samples
            if i < 2:
                print(f"  [Debug] SNR_2500:{snr_2500:5.1f}dB | WPM:{sample_wpms[i]} | Freq:{freqs[i]:5.1f}Hz | Ref:{text:15s} | Hyp:{decoded:15s} | CER:{cer:.4f}")
                
        return cers

def generate_random_text(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length)) + " "   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples", type=int, default=50, help="Samples per SNR point")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_snr_performance.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization")
    parser.add_argument("--fading-speed", type=float, default=0.0)
    parser.add_argument("--min-fading", type=float, default=1.0)
    parser.add_argument("--qrm-prob", type=float, default=0.1)
    parser.add_argument("--impulse-prob", type=float, default=0.001)
    args = parser.parse_args()

    evaluator = PerformanceEvaluator(args.checkpoint)
    dataset = CWDataset() # For phrase generation
    snrs = np.arange(config.EVAL_SNR_MIN, config.EVAL_SNR_MAX, config.EVAL_SNR_STEP)
    
    random_avg_cers = []
    phrase_avg_cers = []

    print(f"Starting evaluation with {args.samples} samples per SNR point...")
    print(f"Settings: Fading={args.fading_speed}, MinFading={args.min_fading}, QRM={args.qrm_prob}, Impulse={args.impulse_prob}")

    for snr in tqdm(snrs):
        # Random 6-char
        random_texts = [generate_random_text(6) for _ in range(args.samples)]
        random_cers = evaluator.evaluate_batch(
            random_texts, snr, random_freq=args.random_freq,
            fading_speed=args.fading_speed, min_fading=args.min_fading,
            qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
        )
        random_avg_cers.append(np.mean(random_cers))

        # Standard Phrases
        phrase_texts = [dataset.generate_phrase() for _ in range(args.samples)]
        phrase_cers = evaluator.evaluate_batch(
            phrase_texts, snr, random_freq=args.random_freq,
            fading_speed=args.fading_speed, min_fading=args.min_fading,
            qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
        )
        phrase_avg_cers.append(np.mean(phrase_cers))

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