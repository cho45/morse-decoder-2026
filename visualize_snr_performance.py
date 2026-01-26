import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
import string
from tqdm import tqdm
from typing import List, Tuple, Dict

from model import StreamingConformer
from data_gen import MorseGenerator, HFChannelSimulator, generate_sample
import config

def decode_ctc_output(logits: torch.Tensor, signal_logits: torch.Tensor) -> str:
    """Greedy decoding with space reconstruction."""
    preds = logits.argmax(axis=-1)  # (T,)
    sig_preds = signal_logits.argmax(axis=-1)  # (T,)

    # CTC greedy decoding
    decoded_indices = []
    decoded_positions = []
    prev = -1
    for t in range(len(preds)):
        idx = preds[t].item()
        if idx != 0 and idx != prev:  # Skip blank(0) and repeats
            decoded_indices.append(idx)
            decoded_positions.append(t)
        prev = idx

    # Space reconstruction
    result = []
    last_pos = 0
    for idx, pos in zip(decoded_indices, decoded_positions):
        # Check for inter-word space (class 3 in 4-class system)
        if any(sig_preds[last_pos:pos] == 3):
            result.append(" ")
        result.append(config.ID_TO_CHAR.get(idx, ""))
        last_pos = pos

    return "".join(result).strip()

def calculate_cer(ref: str, hyp: str) -> float:
    """Calculate Character Error Rate using simple edit distance."""
    # Prosign mapping to single characters for fair evaluation
    prosign_mapping = {ps: chr(i + 1) for i, ps in enumerate(config.PROSIGNS)}
    
    def map_prosigns(text: str) -> str:
        res = text.replace(" ", "")
        for ps, char in prosign_mapping.items():
            res = res.replace(ps, char)
        return res

    ref_mapped = map_prosigns(ref)
    hyp_mapped = map_prosigns(hyp)
    
    if not ref_mapped:
        return 1.0 if hyp_mapped else 0.0
    
    # Simple Levenshtein distance implementation
    n, m = len(ref_mapped), len(hyp_mapped)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_mapped[i-1] == hyp_mapped[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    
    dist = dp[n][m]
    return dist / len(ref_mapped)

class PerformanceEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
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
        
        # Use the same transform as train.py
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0,
            center=False
        ).to(self.device)
        
        self.f_bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
        self.f_bin_end = self.f_bin_start + config.N_BINS
        print(f"Spectrogram Bin Range: {self.f_bin_start} to {self.f_bin_end} ({config.N_BINS} bins)")

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """Linear spectrogram preprocessing matching train.py."""
        with torch.no_grad():
            # Physical Lookahead Padding matching train.py
            lookahead_samples = config.LOOKAHEAD_FRAMES * config.HOP_LENGTH
            padded_waveform = torch.zeros(waveform.size(0) + lookahead_samples, device=self.device)
            padded_waveform[:waveform.size(0)] = waveform.to(self.device)
            
            # Use spec_transform
            spec = self.spec_transform(padded_waveform.unsqueeze(0)) # (B, F, T)
            
            # Slice frequency bins
            spec = spec[:, self.f_bin_start:self.f_bin_end, :]
            
            # Scaling: log1p(x * 100) / 5.0
            mels = torch.log1p(spec * 100.0) / 5.0
            return mels.transpose(1, 2) # (Batch, T, N_BINS)

    def evaluate_batch(self, texts: List[str], snr_db: float, wpm: int = 20, random_freq: bool = False) -> List[float]:
        cers = []
        for text in texts:
            # Generate sample
            freq = random.uniform(config.MIN_FREQ, config.MAX_FREQ) if random_freq else 700.0
            
            # 評価用のデータ生成
            # 学習初期段階のモデルでも解読可能なよう、ジッター等は最小限に抑える
            waveform, _, _, _ = generate_sample(
                text=text, wpm=wpm, snr_db=snr_db, frequency=freq,
                jitter=0.0, weight=1.0, fading_speed=0.0, min_fading=1.0
            )
            
            # Inference
            mels = self.preprocess(waveform)
            with torch.no_grad():
                (ctc, sig, _), _ = self.model(mels)
                decoded = decode_ctc_output(ctc[0], sig[0])
                cer = calculate_cer(text, decoded)
                cers.append(cer)
                # デバッグ用に常に数件表示
                if len(cers) <= 2:
                    print(f"  [Debug] SNR:{snr_db:5.1f}dB | Freq:{freq:5.1f}Hz | Ref:{text:15s} | Hyp:{decoded:15s} | CER:{cer:.4f}")
        return cers

def generate_random_text(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))

from data_gen import CWDataset
def generate_phrase_text(dataset: CWDataset) -> str:
    return dataset.generate_phrase()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples", type=int, default=50, help="Samples per SNR point")
    parser.add_argument("--output", type=str, default="diagnostics/snr_performance_latest.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization (600-800Hz)")
    args = parser.parse_args()

    evaluator = PerformanceEvaluator(args.checkpoint)
    dataset = CWDataset() # For phrase generation
    snrs = np.arange(-18, -2, 1)
    
    random_avg_cers = []
    phrase_avg_cers = []

    if args.random_freq:
        print("  Frequency Randomization: ENABLED (600-800Hz)")
    else:
        print("  Frequency Randomization: DISABLED (Fixed 700Hz)")

    for snr in tqdm(snrs):
        # Random 6-char
        random_texts = [generate_random_text(6) for _ in range(args.samples)]
        random_cers = evaluator.evaluate_batch(random_texts, snr, random_freq=args.random_freq)
        random_avg_cers.append(np.mean(random_cers))

        # Standard Phrases
        phrase_texts = [generate_phrase_text(dataset) for _ in range(args.samples)]
        phrase_cers = evaluator.evaluate_batch(phrase_texts, snr, random_freq=args.random_freq)
        phrase_avg_cers.append(np.mean(phrase_cers))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, random_avg_cers, marker='o', label='Random 6-char (Avg CER)')
    plt.plot(snrs, phrase_avg_cers, marker='s', label='Standard Phrases (Avg CER)')
    
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='CER 10% (Usable)')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='CER 5% (Near Perfect)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"Model Robustness: SNR vs CER (Lower is better)\nCheckpoint: {os.path.basename(args.checkpoint)}")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis() # CER なので上が 0 (良)、下が 1.0 (悪) になるように反転
    plt.ylabel("Character Error Rate (CER) - Top is better")
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()