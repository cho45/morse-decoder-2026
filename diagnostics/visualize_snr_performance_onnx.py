import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import onnxruntime as ort
import multiprocessing
import signal
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from inference_utils import preprocess_waveform, decode_multi_task, calculate_cer
from diagnostics.snr_eval_utils import SyntheticMorseDataset
from data_gen import CWDataset

class ONNXPerformanceEvaluator:
    def __init__(self, model_path: str, chunk_size: int = 40):
        self.chunk_size = chunk_size
        print(f"Loading ONNX model from {model_path}")
        # Dynamic quantization (INT8) often hangs or is unsupported on CUDAExecutionProvider.
        # Use CPU for quantized models, and CUDA for others if available.
        is_quantized = "quantized" in model_path.lower()
        
        available_providers = ort.get_available_providers()
        providers = []
        if not is_quantized and 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        print(f"Model: {os.path.basename(model_path)} | Quantized: {is_quantized} | Providers: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Determine number of layers from inputs
        self.num_layers = 0
        input_names = [i.name for i in self.session.get_inputs()]
        while f"attn_k_{self.num_layers}" in input_names:
            self.num_layers += 1
        print(f"Detected {self.num_layers} layers in ONNX model")

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
        chunk_size = self.chunk_size # Standard chunk size for this model (multiple of 4)
        
        # Get expected input dimension from model
        n_bins = self.session.get_inputs()[0].shape[2]
        
        states = self.init_states(batch_size, n_bins)
        
        # [FIX] Initialize PCEN state with the first frame to avoid warmup issues.
        # Sync with demo/inference.js runFullInference
        if seq_len > 0:
            states['pcen_state'] = mels[:, 0:1, :n_bins].numpy().copy()

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

    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict[float, List[float]]:
        # Map SNR -> List of CERs
        results = defaultdict(list)
        
        for waveforms, actual_texts, wpms, freqs, snrs in tqdm(dataloader, desc="Evaluating", smoothing=0.1):
            # Preprocess on CPU for ONNX
            mels = preprocess_waveform(waveforms, torch.device("cpu"))
            
            logits, signal_logits, boundary_logits = self.run_inference(mels)
            
            bound_probs_batch = torch.sigmoid(boundary_logits).squeeze(-1)

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
    # Use 'spawn' instead of 'fork' to avoid deadlocks with torch/CUDA in subprocesses.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of data loading workers")
    parser.add_argument("--chunk-size", type=int, default=500, help="Streaming chunk size (frames). Larger = faster.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.models):
        print("Error: Number of labels must match number of models")
        return

    labels = args.labels if args.labels else [os.path.basename(m) for m in args.models]
    snrs = np.arange(config.EVAL_SNR_MIN, config.EVAL_SNR_MAX, config.EVAL_SNR_STEP)
    dataset_source = CWDataset()

    plt.figure(figsize=(12, 8))

    for model_path, label in zip(args.models, labels):
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
        
        is_quantized = "quantized" in model_path.lower()
        evaluator = ONNXPerformanceEvaluator(model_path, chunk_size=args.chunk_size)
        
        print(f"Evaluating model: {label} (Quantized: {is_quantized})")
        
        # Standardize on just one mixed dataset or run both separately as in main script?
        # The main script separates them. Let's do a meaningful mix or just one type.
        # Original script did: "Mixed high-density text and phrases"
        # Let's create a single dataset that mixes 50/50 manually or stick to one for simplicity?
        # Better: Run both and average, or just pick 'phrase' which is more realistic.
        # Let's align with the main script and do average of both if possible, or just phrase.
        # For simplicity and consistence with previous ONNX script, let's just do one pass with mixed data?
        # No, SyntheticMorseDataset takes a type. Let's just use 'random' and 'phrase' sequentially and average.
        
        # 1. Random High Density
        random_dataset = SyntheticMorseDataset(
            samples_per_snr=args.samples // 2, snrs=snrs, dataset=dataset_source, wpm=15,
            random_freq=args.random_freq, type='random',
            fading_speed=args.fading_speed, min_fading=args.min_fading,
            qrm_prob=args.qrm_prob, impulse_prob=args.impulse_prob
        )
        random_loader = DataLoader(
            random_dataset, batch_size=args.batch_size, 
            num_workers=args.workers, shuffle=False, drop_last=False,
            persistent_workers=True if args.workers > 0 else False,
            prefetch_factor=2 if args.workers > 0 else None,
            pin_memory=True if torch.cuda.is_available() else False
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
            num_workers=args.workers, shuffle=False, drop_last=False,
            persistent_workers=True if args.workers > 0 else False,
            prefetch_factor=2 if args.workers > 0 else None,
            pin_memory=True if torch.cuda.is_available() else False
        )
        phrase_results = evaluator.evaluate_dataloader(phrase_loader)
        
        # Combine results
        combined_avg_cers = []
        for snr in snrs:
            all_cers = random_results[snr] + phrase_results[snr]
            combined_avg_cers.append(np.mean(all_cers))

        if is_quantized:
            # int8: Green dashed line
            plt.plot(snrs, combined_avg_cers, marker='s', linestyle='--', color='C2', label=f'{label} (ONNX int8)')
        else:
            # fp32: Blue solid line
            plt.plot(snrs, combined_avg_cers, marker='o', linestyle='-', color='C0', label=f'{label} (ONNX fp32)')

    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='CER 10%')
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.3, label='CER 5%')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (in 2500Hz BW) [dB]")
    plt.ylabel("Character Error Rate (CER)")
    plt.title(f"ONNX Model Comparison: SNR_2500 vs CER (Lower is better)")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_yaxis()
    plt.ylabel("Character Error Rate (CER) - Top is better")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()