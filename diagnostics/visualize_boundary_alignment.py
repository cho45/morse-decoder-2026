"""
Visualize Boundary Alignment.
This script generates a synthetic CW signal and plots the waveform, 
signal class labels, and boundary labels to verify the alignment, 
especially for space characters.

Usage:
    python3 diagnostics/visualize_boundary_alignment.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_gen import generate_sample

def main():
    # Parameters
    text = "CQ DE JA1ABC" # Will be "CQ DE JA1ABC " internally
    wpm = 25
    snr = 100 # Clean signal for clear visualization
    
    print(f"Generating sample for text: '{text}' at {wpm} WPM...")
    waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_2500=snr)
    
    # Convert to numpy for plotting
    wf = waveform.numpy()
    sig = signal_labels.numpy()
    bound = boundary_labels.numpy()
    
    # Time axes
    time_wf = np.arange(len(wf)) / config.SAMPLE_RATE
    time_frames = np.arange(len(sig)) * config.HOP_LENGTH / config.SAMPLE_RATE
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Replace space with visual symbol for title and labels
    visual_label = label.replace(' ', '␣')
    
    # 1. Waveform
    axes[0].plot(time_wf, wf, color='gray', alpha=0.3, label='Waveform')
    axes[0].set_title(f"Boundary Alignment Visualization: '{visual_label}' ({wpm} WPM)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    
    # 2. Signal Labels (Classes) with color coding
    # 0: BG, 1: Dit, 2: Dah, 3: Inter-word space
    colors = {
        0: 'white',      # Background
        1: 'lightgreen', # Dit
        2: 'orange',     # Dah
        3: 'lightblue'   # Inter-word space
    }
    class_names = {0: 'Space', 1: 'Dit', 2: 'Dah', 3: 'WordSpace'}
    
    for class_id, color in colors.items():
        mask = (sig == class_id)
        if np.any(mask):
            # Use fill_between for colored background
            axes[1].fill_between(time_frames, 0, 1, where=mask, color=color, alpha=0.5, label=class_names[class_id], step='post')
            axes[0].fill_between(time_frames, -1, 1, where=mask, color=color, alpha=0.1, step='post')

    axes[1].step(time_frames, sig / 3.0, where='post', color='black', linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel("Signal Class")
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', ncol=4)
    
    # 3. Boundary Labels
    axes[2].fill_between(time_frames, 0, bound, color='red', alpha=0.7, label='Boundary Label', step='post')
    axes[2].set_ylabel("Boundary (0/1)")
    axes[2].set_xlabel("Time (sec)")
    axes[2].set_ylim(0, 1.5) # Extra space for text
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Identify characters for each boundary
    # We know the characters are tokens. Space is also a token.
    # Tokens for "CQ DE JA1ABC "
    from data_gen import MorseGenerator
    gen = MorseGenerator()
    tokens = gen.text_to_morse_tokens(label)
    
    # Highlight boundary points and add character labels
    # 境界ラベルが 0 -> 1 に変化する箇所（5フレームの塊の開始）を特定
    diff = np.diff(bound, prepend=0)
    boundary_starts = np.where(diff > 0)[0]
    
    print(f"Tokens: {tokens}")
    print(f"Found {len(boundary_starts)} boundary spikes. Tokens count: {len(tokens)}")
    
    for i, idx in enumerate(boundary_starts):
        t = time_frames[idx]
        axes[0].axvline(x=t, color='red', linestyle='--', alpha=0.5)
        axes[1].axvline(x=t, color='red', linestyle='--', alpha=0.5)
        axes[2].axvline(x=t, color='red', linestyle='--', alpha=0.5)
        
        if i < len(tokens):
            char = tokens[i]
            if char == ' ': char = '␣'
            # Add text label above the spike
            axes[2].text(t, 1.1, char, color='red', fontweight='bold', ha='left', va='bottom', fontsize=12)
    
    # Zoom in on the signal part
    # Find start and end of signal (including spaces)
    non_zero_wf = np.where(np.abs(wf) > 0.01)[0]
    if len(non_zero_wf) > 0:
        start_t = max(0, time_wf[non_zero_wf[0]] - 0.5)
        end_t = min(time_wf[-1], time_wf[non_zero_wf[-1]] + 1.0)
        plt.xlim(start_t, end_t)
    
    # Output file name: basename + .png
    script_name = os.path.basename(__file__)
    output_path = f"diagnostics/{os.path.splitext(script_name)[0]}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()