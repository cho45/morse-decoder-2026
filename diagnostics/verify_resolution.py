import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_gen import generate_sample
import train
import config

def main():
    # Setup
    os.makedirs("diagnostics", exist_ok=True)
    
    # Mock Trainer
    class MockArgs:
        def __init__(self):
            self.lr = 3e-4
            self.freeze_encoder = False
            self.samples_per_epoch = 1000
            self.batch_size = 1
            self.save_dir = "checkpoints"
    
    trainer = train.Trainer(MockArgs())
    
    # Generate Samples for M and K at 20 WPM
    print(f"Config: N_FFT={config.N_FFT}, HOP_LENGTH={config.HOP_LENGTH}, N_MELS={config.N_MELS}")
    print("Generating M and K samples at 20 WPM...")
    
    # M: -- (Dash Dash)
    wf_m, label_m, _ = generate_sample("M", wpm=20, snr_db=100, jitter=0.0)
    
    # K: -.- (Dash Dot Dash)
    wf_k, label_k, _ = generate_sample("K", wpm=20, snr_db=100, jitter=0.0)
    
    # Process
    wf_m_batch = wf_m.unsqueeze(0)
    wf_k_batch = wf_k.unsqueeze(0)
    
    mels_m, _ = trainer.compute_mels_and_lengths(wf_m_batch, torch.tensor([wf_m.shape[0]]))
    mels_k, _ = trainer.compute_mels_and_lengths(wf_k_batch, torch.tensor([wf_k.shape[0]]))
    
    # Analysis: Check for silence gaps
    # Sum energy across frequency bins
    energy_m = mels_m[0].sum(dim=1).cpu().numpy()
    energy_k = mels_k[0].sum(dim=1).cpu().numpy()
    
    # Normalize
    energy_m /= energy_m.max()
    energy_k /= energy_k.max()
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # M
    ax_wf_m = axes[0, 0]
    ax_wf_m.plot(wf_m.numpy())
    ax_wf_m.set_title("Waveform: M (Dash Dash)")
    ax_wf_m.set_xlabel("Samples")
    
    ax_mel_m = axes[0, 1]
    mel_data_m = mels_m[0].transpose(0, 1).cpu().numpy()
    im_m = ax_mel_m.imshow(mel_data_m, aspect='auto', origin='lower', interpolation='nearest')
    ax_mel_m.set_title(f"Mel Spec: M")
    ax_mel_m.set_xlabel("Frames")
    fig.colorbar(im_m, ax=ax_mel_m)
    
    # K
    ax_wf_k = axes[1, 0]
    ax_wf_k.plot(wf_k.numpy())
    ax_wf_k.set_title("Waveform: K (Dash Dot Dash)")
    ax_wf_k.set_xlabel("Samples")
    
    ax_mel_k = axes[1, 1]
    mel_data_k = mels_k[0].transpose(0, 1).cpu().numpy()
    im_k = ax_mel_k.imshow(mel_data_k, aspect='auto', origin='lower', interpolation='nearest')
    ax_mel_k.set_title(f"Mel Spec: K")
    ax_mel_k.set_xlabel("Frames")
    fig.colorbar(im_k, ax=ax_mel_k)
    
    plt.tight_layout()
    output_path = "diagnostics/resolution_check.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    
    # Numerical Check
    print("\nNumerical Analysis:")
    print(f"M Length (frames): {len(energy_m)}")
    print(f"K Length (frames): {len(energy_k)}")
    
    # Count distinct peaks in energy (simple thresholding)
    def count_peaks(energy, threshold=0.5):
        is_on = energy > threshold
        changes = np.diff(is_on.astype(int))
        # Number of rising edges
        return np.sum(changes == 1) + (1 if is_on[0] else 0)

    peaks_m = count_peaks(energy_m)
    peaks_k = count_peaks(energy_k)
    
    print(f"Detected Peaks (Energy > 0.5): M={peaks_m}, K={peaks_k}")
    print(f"Expected Peaks: M=2 (Dash, Dash), K=3 (Dash, Dot, Dash)")
    
    if peaks_m == 2 and peaks_k == 3:
        print("SUCCESS: Resolution is sufficient to distinguish pulses.")
    else:
        print("FAILURE: Resolution insufficient or thresholding failed.")

if __name__ == "__main__":
    main()
