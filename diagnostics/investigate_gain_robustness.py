import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import StreamingConformer
from data_gen import generate_sample
import config
import argparse

def investigate_gain(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = StreamingConformer().to(device)
    
    # Handle size mismatch due to new PCEN parameters
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    
    # Check if PCEN parameters were loaded
    pcen_keys = [k for k in pretrained_dict.keys() if 'pcen' in k]
    if pcen_keys:
        print(f"Loaded PCEN parameters from checkpoint: {pcen_keys}")
    else:
        print("PCEN parameters not found in checkpoint or shape mismatch. Using default values.")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    # Display PCEN parameters
    with torch.no_grad():
        s = torch.exp(model.pcen.log_s).mean().item()
        alpha = torch.exp(model.pcen.log_alpha).mean().item()
        delta = torch.exp(model.pcen.log_delta).mean().item()
        r = torch.exp(model.pcen.log_r).mean().item()
        print(f"PCEN Parameters (Mean): s={s:.4f}, alpha={alpha:.4f}, delta={delta:.4f}, r={r:.4f}")

    # 2. Spectrogram Transform (same as in train.py)
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        power=2.0,
        center=False
    ).to(device)
    bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
    bin_end = bin_start + config.N_BINS

    # 3. Hooks to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    model.subsampling.conv.register_forward_hook(get_activation('conv_out'))
    model.subsampling.out_linear.register_forward_hook(get_activation('subsampling_out'))

    # 4. Generate Test Samples with different gains
    test_text = "CQ CQ CQ"
    gains_db = [0, -20, -40, -60]
    
    fig, axes = plt.subplots(len(gains_db), 3, figsize=(15, 4 * len(gains_db)))
    
    for i, gain_db in enumerate(gains_db):
        print(f"Testing Gain: {gain_db} dB")
        
        # Generate clean sample first, then apply gain
        waveform, label, signal_labels, boundary_labels = generate_sample(
            test_text, wpm=20, snr_2500=100, frequency=700.0
        )
        
        # Apply gain
        gain = 10 ** (gain_db / 20)
        waveform = waveform * gain
        
        # Process through spectrogram
        x = waveform.unsqueeze(0).to(device)
        spec = spec_transform(x)
        spec = spec[:, bin_start:bin_end, :]
        
        # New logic: raw spec passed to model, PCEN handles it
        mels = spec.transpose(1, 2) # (B, T, F)
        
        # Forward through model with PCEN state
        with torch.no_grad():
            # Capture PCEN output manually for visualization
            input_scaled = mels * model.input_scale
            pcen_out, _ = model.pcen(input_scaled)
            activations['pcen_out'] = pcen_out.detach()
            
            # Full model forward
            (logits, sig_logits, bound_logits), _ = model(mels)
        
        # Plotting
        ax_in = axes[i, 0]
        im_in = ax_in.imshow(activations['pcen_out'][0].cpu().T, aspect='auto', origin='lower')
        ax_in.set_title(f"PCEN Out (Gain {gain_db}dB)\nMax: {activations['pcen_out'].max():.4f}")
        fig.colorbar(im_in, ax=ax_in)

        ax_conv = axes[i, 1]
        conv_out = activations['conv_out'][0].mean(dim=0) # Mean across channels for visualization
        im_conv = ax_conv.imshow(conv_out.cpu(), aspect='auto', origin='lower')
        ax_conv.set_title(f"Conv Out (Mean Ch)\nMax: {activations['conv_out'].max():.4f}")
        fig.colorbar(im_conv, ax=ax_conv)

        ax_sub = axes[i, 2]
        im_sub = ax_sub.imshow(activations['subsampling_out'][0].cpu().T, aspect='auto', origin='lower')
        ax_sub.set_title(f"Subsampling Out\nMax: {activations['subsampling_out'].max():.4f}")
        fig.colorbar(im_sub, ax=ax_sub)

        # Print statistics
        print(f"  PCEN Out    | Max: {activations['pcen_out'].max():.6f}, Mean: {activations['pcen_out'].mean():.6f}")
        print(f"  Conv Out    | Max: {activations['conv_out'].max():.6f}, Mean: {activations['conv_out'].mean():.6f}")
        print(f"  Sub Out     | Max: {activations['subsampling_out'].max():.6f}, Mean: {activations['subsampling_out'].mean():.6f}")
        
        # Check for zero activations
        zero_ratio = (activations['conv_out'] == 0).float().mean().item()
        print(f"  Conv Zero Ratio: {zero_ratio:.2%}")

    plt.tight_layout()
    output_png = "diagnostics/investigate_gain_robustness.png"
    plt.savefig(output_png)
    print(f"Saved visualization to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_614.pt")
    args = parser.parse_args()
    
    investigate_gain(args.checkpoint)