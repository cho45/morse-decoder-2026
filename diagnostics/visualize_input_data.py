"""
指定した文字列とWPMでCW信号を生成し、スペクトログラムとSignalラベルを可視化するスクリプト。

使い方:
    python3 diagnostics/visualize_input_data.py --text "CQ DE JE1TRV" --wpm 25 --snr 20
    
出力:
    diagnostics/visualize_input_data.png
"""
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_gen import generate_sample

def plot_spectrogram(ax, waveform, title):
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        power=2.0,
        center=False
    )
    spec = spec_transform(waveform)
    # Use log scale for visibility
    spec_db = 10 * torch.log10(spec + 1e-9)
    
    # We only care about the CW frequency range (500-900Hz)
    bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
    bin_end = bin_start + config.N_BINS
    spec_crop = spec_db[bin_start:bin_end, :]
    
    # extent=[left, right, bottom, top]
    duration = len(waveform) / config.SAMPLE_RATE
    im = ax.imshow(spec_crop.numpy(), aspect='auto', origin='lower', extent=[0, duration, config.F_MIN, config.F_MAX])
    ax.set_title(title)
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")
    return im

def main():
    parser = argparse.ArgumentParser(description="Visualize generated CW signal.")
    parser.add_argument("--text", type=str, default="CQ DE JE1TRV", help="Text to generate")
    parser.add_argument("--wpm", type=int, default=20, help="Words per minute")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR in dB")
    parser.add_argument("--output", type=str, default="diagnostics/visualize_input_data.png", help="Output image path")
    
    args = parser.parse_args()
    
    # Generate sample
    print(f"Generating sample: Text='{args.text}', WPM={args.wpm}, SNR={args.snr}dB")
    
    # generate_sample returns: waveform, text, signal_labels, boundary_labels
    waveform, _, signal_labels, boundary_labels = generate_sample(
        text=args.text,
        wpm=args.wpm,
        snr_2500=args.snr,
        sample_rate=config.SAMPLE_RATE,
        max_duration=10.0 # Allow longer duration for visualization
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    duration = len(waveform) / config.SAMPLE_RATE
    title = f"Text: {args.text} | WPM: {args.wpm} | SNR: {args.snr}dB"
    plot_spectrogram(ax, waveform, title)
    
    # Overlay signal labels
    # 0: Background (none), 1: Dit (Yellow), 2: Dah (Cyan), 3: WordSpace (Red)
    num_frames = len(signal_labels)
    times = np.linspace(0, duration, num_frames)
    
    colors = {1: 'yellow', 2: 'cyan', 3: 'red'}
    labels = {1: 'Dit', 2: 'Dah', 3: 'WordSpace'}
    
    # Create dummy plots for legend
    legend_handles = []
    for cid, color in colors.items():
        # Create a dummy patch for legend
        patch = mpatches.Patch(color=color, alpha=0.5, label=labels[cid])
        legend_handles.append(patch)

    for cid, color in colors.items():
        mask = (signal_labels.numpy() == cid)
        if mask.any():
            # Use fill_between for actual visualization
            # We use a small strip at the bottom of the frequency range
            # Adjust mapping to match time axis correctly
            # times array length matches mask length
            ax.fill_between(times, config.F_MIN, config.F_MIN + 50, where=mask,
                            color=color, alpha=0.5, linewidth=0)
    
    # Overlay Boundary labels
    if boundary_labels.sum() > 0:
        boundary_mask = boundary_labels.numpy() > 0.5
        ax.vlines(times[boundary_mask], config.F_MIN, config.F_MAX, color='white', alpha=0.5, linestyles='--', label='Boundary')
        legend_handles.append(plt.Line2D([0], [0], color='white', linestyle='--', label='Boundary'))

    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()