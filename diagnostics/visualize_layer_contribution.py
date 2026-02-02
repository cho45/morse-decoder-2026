import os
import sys
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
from model import StreamingConformer
from data_gen import CWDataset

import re

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    
    def extract_epoch(filename):
        # Extract the last number in the filename
        match = re.search(r'epoch_(\d+)', filename)
        if match:
            return int(match.group(1))
        return -1

    # Sort by epoch number
    latest_checkpoint = max(checkpoints, key=extract_epoch)
    return latest_checkpoint

class ContributionTracker:
    def __init__(self, model):
        self.hooks = []
        self.data = {}  # {layer_idx: {module_name: {'input_norm': [], 'output_norm': []}}}
        self.model = model
        self.register_hooks()

    def _get_hook(self, layer_idx, module_name, is_residual_scaled=False, is_tuple_output=False):
        def hook(module, input, output):
            # input is a tuple
            in_tensor = input[0]
            
            if is_tuple_output:
                out_tensor = output[0]
            else:
                out_tensor = output

            # Calculate norms (average over batch and sequence)
            # (B, T, D) -> scalar
            # Use float64 for accumulation to avoid overflow/underflow in stats, though unlikely here
            in_norm = torch.norm(in_tensor, p=2, dim=-1).mean().item()
            
            if is_residual_scaled:
                out_tensor = out_tensor * 0.5
                
            out_norm = torch.norm(out_tensor, p=2, dim=-1).mean().item()

            if layer_idx not in self.data:
                self.data[layer_idx] = {}
            if module_name not in self.data[layer_idx]:
                self.data[layer_idx][module_name] = {'input_norm': [], 'output_norm': []}
            
            self.data[layer_idx][module_name]['input_norm'].append(in_norm)
            self.data[layer_idx][module_name]['output_norm'].append(out_norm)
            
        return hook

    def register_hooks(self):
        for i, layer in enumerate(self.model.layers):
            # FF1
            # Input: ln_ff1 input is the block input
            # We hook ln_ff1 to capture the input to the first normalization
            self.hooks.append(layer.ln_ff1.register_forward_hook(
                self._get_hook(i, 'FF1_in', is_residual_scaled=False)
            ))
            # Output: ff1 output (before 0.5 scaling)
            self.hooks.append(layer.ff1.register_forward_hook(
                self._get_hook(i, 'FF1_out', is_residual_scaled=True)
            ))

            # Attention
            # Input: ln_attn input (LayerNorm returns Tensor)
            self.hooks.append(layer.ln_attn.register_forward_hook(
                self._get_hook(i, 'Attn_in', is_residual_scaled=False, is_tuple_output=False)
            ))
            # Output: attn output (tuple[0])
            self.hooks.append(layer.attn.register_forward_hook(
                self._get_hook(i, 'Attn_out', is_residual_scaled=False, is_tuple_output=True)
            ))

            # Convolution
            # Input: conv module input.
            # Note: We are hooking the module itself, so we must handle its output correctly even if we only care about input.
            # ConformerConvModule returns a tuple.
            self.hooks.append(layer.conv.register_forward_hook(
                self._get_hook(i, 'Conv_in', is_residual_scaled=False, is_tuple_output=True)
            ))
            # Output: conv output (tuple[0])
            self.hooks.append(layer.conv.register_forward_hook(
                self._get_hook(i, 'Conv_out', is_residual_scaled=False, is_tuple_output=True)
            ))

            # FF2
            # Input: ln_ff2 input
            self.hooks.append(layer.ln_ff2.register_forward_hook(
                self._get_hook(i, 'FF2_in', is_residual_scaled=False)
            ))
            # Output: ff2 output (before 0.5 scaling)
            self.hooks.append(layer.ff2.register_forward_hook(
                self._get_hook(i, 'FF2_out', is_residual_scaled=True)
            ))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    ckpt_path = get_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoint found in checkpoints/")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = StreamingConformer(
        n_mels=config.N_BINS, 
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL, 
        n_head=config.N_HEAD, 
        num_layers=config.NUM_LAYERS, 
        kernel_size=config.KERNEL_SIZE, 
        dropout=0.0 # Disable dropout for deterministic measurement
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare Data
    print("Generating test data...")
    dataset = CWDataset(
        num_samples=16,
        min_snr_2500=10.0,
        max_snr_2500=20.0
    )
    
    # Create a batch
    batch_size = 16
    specs = []
    
    # Need to compute spectrogram manually as CWDataset returns waveform
    window = torch.hann_window(config.N_FFT).to(device)
    
    for _ in range(batch_size):
        waveform, _, _, _, _, _ = dataset[0]
        waveform = waveform.to(device)
        
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            window=window,
            return_complex=True
        )
        # Power spectrogram
        spec = torch.abs(stft).pow(2)
        
        # Extract bins
        f_min_idx = int(config.F_MIN / (config.SAMPLE_RATE / config.N_FFT))
        f_max_idx = f_min_idx + config.N_BINS
        spec = spec[f_min_idx:f_max_idx, :] # (F, T)
        spec = spec.transpose(0, 1) # (T, F)
        
        # Log scaling (simple version, actual training might use different scaling but PCEN handles it)
        # However, model expects linear power spec for PCEN usually, or log spec?
        # Checking model.py: PCEN takes input. Usually PCEN takes power or magnitude.
        # Let's check config or training loop.
        # train.py uses:
        # spec = torch.stft(..., return_complex=True)
        # spec = torch.abs(spec)
        # spec = spec[f_min_idx:f_max_idx, :]
        # spec = spec.transpose(0, 1)
        # So it uses Magnitude, not Power.
        
        spec = torch.abs(stft)
        spec = spec[f_min_idx:f_max_idx, :]
        spec = spec.transpose(0, 1)
        
        specs.append(spec)
    
    x = torch.stack(specs).to(device) # (B, T, F)
    
    # Initialize states
    states = model.get_initial_states(batch_size, device)

    # Track contributions
    tracker = ContributionTracker(model)
    
    print("Running inference...")
    with torch.no_grad():
        model(x, states)
    
    tracker.remove_hooks()

    # Analyze results
    print("Analyzing contributions...")
    
    layers = sorted(tracker.data.keys())
    modules = ['FF1', 'Attn', 'Conv', 'FF2']
    
    contributions = {m: [] for m in modules}
    
    print(f"{'Layer':<6} {'FF1':<10} {'Attn':<10} {'Conv':<10} {'FF2':<10}")
    
    for l in layers:
        layer_data = tracker.data[l]
        
        # FF1
        ff1_in = np.mean(layer_data['FF1_in']['input_norm'])
        ff1_out = np.mean(layer_data['FF1_out']['output_norm']) # Already scaled by 0.5 in hook
        ratio_ff1 = ff1_out / ff1_in
        contributions['FF1'].append(ratio_ff1)
        
        # Attn
        attn_in = np.mean(layer_data['Attn_in']['input_norm'])
        attn_out = np.mean(layer_data['Attn_out']['output_norm'])
        ratio_attn = attn_out / attn_in
        contributions['Attn'].append(ratio_attn)
        
        # Conv
        conv_in = np.mean(layer_data['Conv_in']['input_norm'])
        conv_out = np.mean(layer_data['Conv_out']['output_norm'])
        ratio_conv = conv_out / conv_in
        contributions['Conv'].append(ratio_conv)
        
        # FF2
        ff2_in = np.mean(layer_data['FF2_in']['input_norm'])
        ff2_out = np.mean(layer_data['FF2_out']['output_norm']) # Already scaled by 0.5 in hook
        ratio_ff2 = ff2_out / ff2_in
        contributions['FF2'].append(ratio_ff2)
        
        print(f"{l+1:<6} {ratio_ff1:<10.4f} {ratio_attn:<10.4f} {ratio_conv:<10.4f} {ratio_ff2:<10.4f}")

    # Plotting
    x_axis = np.arange(1, len(layers) + 1)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, module in enumerate(modules):
        offset = (i - 1.5) * width
        ax.bar(x_axis + offset, contributions[module], width, label=module)

    ax.set_ylabel('Relative Contribution (||Update|| / ||Input||)')
    ax.set_xlabel('Layer Index')
    ax.set_title('Conformer Layer Contribution Analysis')
    ax.set_xticks(x_axis)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = os.path.join('diagnostics', 'visualize_layer_contribution.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()