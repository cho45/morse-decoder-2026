import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import string
from tqdm import tqdm
from typing import List, Tuple

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

    def evaluate_batch(self, texts: List[str], snr_2500: float, wpm: int = 15, random_freq: bool = False,
                       fading_speed: float = 0.0, min_fading: float = 1.0,
                       qrm_prob: float = 0.0, impulse_prob: float = 0.0) -> List[float]:
        waveforms = []
        sample_wpms = []
        freqs = []
        actual_texts = []
        
        # 1. Generate all waveforms in CPU loop
        for text in texts:
            freq = random.uniform(config.MIN_FREQ, config.MAX_FREQ) if random_freq else 700.0
            sample_wpm = wpm
            
            # generate_sample will handle text reconstruction/truncation
            waveform, actual_text, _, _ = generate_sample(
                text=text, wpm=sample_wpm, snr_2500=snr_2500, frequency=freq,
                jitter=0.0, weight=1.0, fading_speed=fading_speed, min_fading=min_fading,
                qrm_prob=qrm_prob, impulse_prob=impulse_prob
            )
            actual_texts.append(actual_text)
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
        for i, text in enumerate(actual_texts):
            decoded, _ = decode_multi_task(ctc_batch[i], sig_batch[i], bound_probs_batch[i])
            cer = calculate_cer(text, decoded)
            cers.append(cer)
            
            # Debug display for first few samples
            if i < 2:
                print(f"  [Debug] SNR_2500:{snr_2500:5.1f}dB | WPM:{sample_wpms[i]} | Freq:{freqs[i]:5.1f}Hz | Ref:{text:15s} | Hyp:{decoded:15s} | CER:{cer:.4f}")
                
        return cers

def generate_random_text(wpm: int = 15) -> str:
    """Generate random text that fits in 10s at given WPM with high density."""
    gen = MorseGenerator()
    # Target about 80% of 10s
    max_chars = gen.estimate_max_chars_for_wpm(wpm, target_frames=800)
    chars = string.ascii_uppercase + string.digits
    text = "".join(random.choices(chars, k=max_chars))
    # Add some spaces
    text_with_spaces = ""
    for c in text:
        text_with_spaces += c
        if random.random() < 0.2:
            text_with_spaces += " "
    return text_with_spaces.strip() + " "

def generate_packed_phrase(dataset: CWDataset, wpm: int = 15) -> str:
    """Generate multiple phrases concatenated to fit in 10s."""
    gen = MorseGenerator()
    max_duration = 10.0
    text = dataset.generate_phrase()
    
    for _ in range(3):
        next_phrase = dataset.generate_phrase()
        if gen.estimate_duration(text + " " + next_phrase, wpm) < max_duration - 1.0:
            text += " " + next_phrase
        else:
            break
    return text.strip() + " "

def get_evaluation_texts(num_samples: int, dataset: CWDataset, wpm: int = 15) -> Tuple[List[str], List[str]]:
    """Get two lists of texts: random and phrase based."""
    random_texts = [generate_random_text(wpm=wpm) for _ in range(num_samples)]
    phrase_texts = [generate_packed_phrase(dataset, wpm=wpm) for _ in range(num_samples)]
    return random_texts, phrase_texts

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
    args = parser.parse_args()

    evaluator = PerformanceEvaluator(args.checkpoint)
    dataset = CWDataset() # For phrase generation
    snrs = np.arange(config.EVAL_SNR_MIN, config.EVAL_SNR_MAX, config.EVAL_SNR_STEP)
    
    random_avg_cers = []
    phrase_avg_cers = []

    print(f"Starting evaluation with {args.samples} samples per SNR point...")
    print(f"Settings: Fading={args.fading_speed}, MinFading={args.min_fading}, QRM={args.qrm_prob}, Impulse={args.impulse_prob}")

    for snr in tqdm(snrs):
        random_texts, phrase_texts = get_evaluation_texts(args.samples, dataset, wpm=15)
        
        # Random high-density text
        random_cers = evaluator.evaluate_batch(
            random_texts, snr, wpm=15, random_freq=args.random_freq,
            fading_speed=args.fading_speed, min_fading=args.min_fading,
            qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
        )
        random_avg_cers.append(np.mean(random_cers))

        # Packed Phrases
        phrase_cers = evaluator.evaluate_batch(
            phrase_texts, snr, wpm=15, random_freq=args.random_freq,
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