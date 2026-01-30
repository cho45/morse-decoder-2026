"""
カリキュラム学習の各フェーズにおけるデータ生成結果を可視化するスクリプト。
各フェーズのパラメータ（SNR、フェージング、ドリフト、AGCなど）が適用されたスペクトログラムを表示し、
正解ラベル（Dit/Dah/Space）と境界ラベルをオーバーレイします。

使い方:
    python3 diagnostics/visualize_curriculum_phases.py
    
出力:
    diagnostics/visualize_curriculum_phases.png
"""
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum import CurriculumManager
from data_gen import CWDataset
import config

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
    
    # 正確な時間軸の計算
    num_frames = spec_crop.shape[1]
    duration = num_frames * config.HOP_LENGTH / config.SAMPLE_RATE
    
    im = ax.imshow(spec_crop.numpy(), aspect='auto', origin='lower', extent=[0, duration, config.F_MIN, config.F_MAX])
    ax.set_title(title)
    ax.set_ylabel("Freq (Hz)")
    return im, duration

def visualize_phases():
    cm = CurriculumManager()
    dataset = CWDataset(num_samples=1)
    
    # Select key phases to visualize
    # Practical_1 (Fading), Practical_1_Drift, Practical_1_AGC, Practical_2 (8dB), Negative_2 (-10dB), Extreme (-15dB)
    target_phases = [
        "Char_1_1", # Clean
        "Practical_1", # Fading
        "Practical_1_Drift", # +Drift
        "Practical_1_AGC", # +AGC
        "Practical_2", # SNR 8-18
        "Negative_2", # SNR -10-0
        "Extreme" # SNR -15
    ]
    
    phases_to_plot = []
    for i in range(1, cm.get_max_phase() + 1):
        p = cm.get_phase(i)
        if p.name in target_phases:
            phases_to_plot.append((i, p))
            
    fig, axes = plt.subplots(len(phases_to_plot), 1, figsize=(12, 3 * len(phases_to_plot)), sharex=True)
    if len(phases_to_plot) == 1: axes = [axes]
    
    for ax, (idx, p) in zip(axes, phases_to_plot):
        # Apply phase settings to dataset
        dataset.min_snr_2500 = p.min_snr_2500
        dataset.max_snr_2500 = p.max_snr_2500
        dataset.jitter_max = p.jitter
        dataset.weight_var = p.weight_var
        dataset.fading_speed_min = p.fading_speed[0]
        dataset.fading_speed_max = p.fading_speed[1]
        dataset.min_fading = p.min_fading
        dataset.drift_prob = p.drift_prob
        dataset.agc_prob = p.agc_prob
        dataset.qrm_prob = p.qrm_prob
        dataset.impulse_prob = p.impulse_prob
        dataset.qrn_prob = p.qrn_prob
        dataset.clipping_prob = p.clipping_prob
        dataset.min_gain_db = p.min_gain_db
        dataset.phrase_prob = 1.0 # Force phrase for better visual
        
        # Generate one sample
        waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[0]
        
        title = f"Phase {idx}: {p.name} | SNR_2500={p.min_snr_2500:.1f}-{p.max_snr_2500:.1f}\nDrift={p.drift_prob} | AGC={p.agc_prob} | QRM={p.qrm_prob} | Impulse={p.impulse_prob}"
        _, duration = plot_spectrogram(ax, waveform, title)
        
        # Overlay signal labels (dots/dashes) using colors
        # 0: Background (none), 1: Dit (Yellow), 2: Dah (Cyan), 3: WordSpace (Red)
        # 正確な時間軸を使用
        times = np.linspace(0, duration, len(signal_labels))
        colors = {1: 'yellow', 2: 'cyan', 3: 'red'}
        for cid, color in colors.items():
            # Use numpy conversion for where parameter
            mask = (signal_labels.numpy() == cid)
            if mask.any():
                ax.fill_between(times, config.F_MIN, config.F_MIN + 30, where=mask,
                                color=color, alpha=0.8, label=f'Class {cid}')
        
        # Overlay Boundary labels as thin white lines
        if boundary_labels.sum() > 0:
            ax.vlines(times[boundary_labels > 0.5], config.F_MIN, config.F_MAX, color='white', alpha=0.3, linestyles='--')

    plt.xlabel("Time (s)")
    plt.tight_layout()
    output_path = "diagnostics/visualize_curriculum_phases.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_phases()