import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import string
import onnxruntime as ort
from tqdm import tqdm
from typing import List, Tuple, Dict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_gen import generate_sample, CWDataset, MorseGenerator
from inference_utils import preprocess_waveform, decode_multi_task, calculate_cer

def generate_random_text(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))

class ONNXPerformanceEvaluator:
    def __init__(self, model_path: str):
        print(f"Loading ONNX model from {model_path}")
        # Use CPU provider for maximum compatibility
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Determine number of layers from inputs
        self.num_layers = 0
        input_names = [i.name for i in self.session.get_inputs()]
        while f"attn_k_{self.num_layers}" in input_names:
            self.num_layers += 1
        print(f"Detected {self.num_layers} layers in ONNX model")

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0,
            center=False
        )
        self.f_bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
        self.f_bin_end = self.f_bin_start + config.N_BINS
        self.gen = MorseGenerator()

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """Standardized preprocessing matching inference_utils.py."""
        # inference_utils.preprocess_waveform already handles padding and cropping
        device = torch.device("cpu")
        return preprocess_waveform(waveform, device)

    def init_states(self, batch_size: int = 1, n_bins: int = 16):
        d_k = config.D_MODEL // config.N_HEAD
        states = {
            'pcen_state': np.zeros((batch_size, 1, n_bins), dtype=np.float32),
            'sub_cache': np.zeros((batch_size, 1, 2, n_bins), dtype=np.float32)
        }
        for i in range(self.num_layers):
            states[f'attn_k_{i}'] = np.zeros((batch_size, config.N_HEAD, 0, d_k), dtype=np.float32)
            states[f'attn_v_{i}'] = np.zeros((batch_size, config.N_HEAD, 0, d_k), dtype=np.float32)
            states[f'offset_{i}'] = np.array(0, dtype=np.int64)
            states[f'conv_cache_{i}'] = np.zeros((batch_size, config.D_MODEL, config.KERNEL_SIZE - 1), dtype=np.float32)
        return states

    def run_inference(self, mels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run streaming inference on full mels by chunking."""
        batch_size = mels.size(0)
        seq_len = mels.size(1)
        chunk_size = 40 # Standard chunk size for this model (multiple of 4)
        
        # Get expected input dimension from model
        n_bins = self.session.get_inputs()[0].shape[2]
        
        states = self.init_states(batch_size, n_bins)
        
        all_logits = []
        all_signal_logits = []
        all_boundary_logits = []
        
        for i in range(0, seq_len, chunk_size):
            chunk_torch = mels[:, i:i+chunk_size, :]
            
            # Pad frequency bins if necessary
            if chunk_torch.size(2) < n_bins:
                pad_freq = n_bins - chunk_torch.size(2)
                chunk_torch = torch.nn.functional.pad(chunk_torch, (0, pad_freq))
            elif chunk_torch.size(2) > n_bins:
                chunk_torch = chunk_torch[:, :, :n_bins]
                
            chunk = chunk_torch.numpy()
            
            # Ensure chunk size is multiple of 4 for subsampling
            orig_len = chunk.shape[1]
            if orig_len % 4 != 0:
                pad_len = 4 - (orig_len % 4)
                chunk = np.pad(chunk, ((0,0), (0, pad_len), (0,0)), mode='constant')
            
            inputs = {'x': chunk}
            inputs.update(states)
            
            outputs = self.session.run(None, inputs)
            
            # Output order: (logits, signal_logits, boundary_logits), (new_pcen_state, new_sub_cache, new_layer_states)
            logits, signal_logits, boundary_logits = outputs[0], outputs[1], outputs[2]
            
            # Trim padding from logits (subsampled)
            if orig_len % 4 != 0:
                valid_len = (orig_len + config.SUBSAMPLING_RATE - 1) // config.SUBSAMPLING_RATE
                logits = logits[:, :valid_len, :]
                signal_logits = signal_logits[:, :valid_len, :]
                boundary_logits = boundary_logits[:, :valid_len, :]

            all_logits.append(logits)
            all_signal_logits.append(signal_logits)
            all_boundary_logits.append(boundary_logits)
            
            # Update states
            states['pcen_state'] = outputs[3]
            states['sub_cache'] = outputs[4]
            for l in range(self.num_layers):
                states[f'attn_k_{l}'] = outputs[5 + l*4]
                states[f'attn_v_{l}'] = outputs[5 + l*4 + 1]
                states[f'offset_{l}'] = outputs[5 + l*4 + 2]
                states[f'conv_cache_{l}'] = outputs[5 + l*4 + 3]
                
        full_logits = np.concatenate(all_logits, axis=1)
        full_signal_logits = np.concatenate(all_signal_logits, axis=1)
        full_boundary_logits = np.concatenate(all_boundary_logits, axis=1)
        
        # Trim back to original length (subsampled)
        expected_len = (seq_len + config.SUBSAMPLING_RATE - 1) // config.SUBSAMPLING_RATE
        full_logits = full_logits[:, :expected_len, :]
        full_signal_logits = full_signal_logits[:, :expected_len, :]
        full_boundary_logits = full_boundary_logits[:, :expected_len, :]
        
        return torch.from_numpy(full_logits), torch.from_numpy(full_signal_logits), torch.from_numpy(full_boundary_logits)

    def evaluate_batch(self, texts: List[str], snr_db: float, wpm: int = 20, random_freq: bool = False,
                       fading_speed: float = 0.0, min_fading: float = 1.0,
                       qrm_prob: float = 0.1, impulse_prob: float = 0.001) -> List[float]:
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
                jitter=0.0, weight=1.0, fading_speed=fading_speed, min_fading=min_fading,
                qrm_prob=qrm_prob, impulse_prob=impulse_prob
            )
            
            mels = self.preprocess(waveform)
            logits, signal_logits, boundary_logits = self.run_inference(mels)
            
            # Unified Decoding
            bound_probs = torch.sigmoid(boundary_logits[0]).squeeze(-1)
            decoded, _ = decode_multi_task(logits[0], signal_logits[0], bound_probs)
            
            cer = calculate_cer(text, decoded)
            cers.append(cer)
        return cers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True, help="List of ONNX model paths")
    parser.add_argument("--labels", type=str, nargs='+', help="Labels for the models in the plot")
    parser.add_argument("--samples", type=int, default=30, help="Samples per SNR point")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_snr_performance_onnx.png")
    parser.add_argument("--random-freq", action="store_true", help="Enable frequency randomization")
    parser.add_argument("--fading-speed", type=float, default=0.0)
    parser.add_argument("--min-fading", type=float, default=1.0)
    parser.add_argument("--qrm-prob", type=float, default=0.1)
    parser.add_argument("--impulse-prob", type=float, default=0.001)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.models):
        print("Error: Number of labels must match number of models")
        return

    labels = args.labels if args.labels else [os.path.basename(m) for m in args.models]
    snrs = np.arange(-18, 2, 1)

    
    plt.figure(figsize=(12, 8))
    
    for model_path, label in zip(args.models, labels):
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
            
        evaluator = ONNXPerformanceEvaluator(model_path)
        
        avg_cers = []
        print(f"Evaluating model: {label}")
        
        for snr in tqdm(snrs):
            texts = [generate_random_text(6) for _ in range(args.samples)]
            cers = evaluator.evaluate_batch(
                texts, snr, random_freq=args.random_freq,
                fading_speed=args.fading_speed, min_fading=args.min_fading,
                qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
            )
            avg_cer = np.mean(cers)
            avg_cers.append(avg_cer)
            print(f"  SNR: {snr:3d}dB | Avg CER: {avg_cer:.4f}")

        plt.plot(snrs, avg_cers, marker='o', label=f'{label} (Avg CER)')

    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='CER 10%')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.3, label='CER 5%')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"ONNX Model Comparison: SNR vs CER (Lower is better)")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis()
    plt.ylabel("Character Error Rate (CER) - Top is better")
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()