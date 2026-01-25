import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from data_gen import CWDataset, generate_sample
import config
import train

def main():
    output_path = "diagnostics/boundary_check.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize Trainer to use its preprocessing logic
    class MockArgs:
        def __init__(self):
            self.lr = 3e-4
            self.freeze_encoder = False
            self.samples_per_epoch = 1000
            self.batch_size = 1
            self.curriculum_phase = 1
            self.save_dir = "checkpoints"
            self.d_model = config.D_MODEL
            self.n_head = config.N_HEAD
            self.num_layers = config.NUM_LAYERS
            self.dropout = config.DROPOUT
    
    trainer = train.Trainer(MockArgs())
    
    # Generate a specific sample for clear visualization
    text = "KM"
    wpm = 20
    waveform, label, signal_labels, boundary_labels = generate_sample(text, wpm=wpm, snr_db=100)
    
    # Preprocess (Mel and Subsampling alignment)
    wf_batch = waveform.unsqueeze(0)
    mels, input_lengths = trainer.compute_mels_and_lengths(wf_batch, torch.tensor([waveform.shape[0]]))
    
    # Label downsampling logic (from train.py collate_fn)
    s_downsampled = signal_labels[::config.SUBSAMPLING_RATE]
    b_downsampled = boundary_labels[::config.SUBSAMPLING_RATE]
    
    # Truncate to match model output length
    out_len = input_lengths[0].item()
    s_downsampled = s_downsampled[:out_len]
    b_downsampled = b_downsampled[:out_len]

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=False)
    
    # 1. Waveform and Raw Labels
    ax0 = axes[0]
    ax0.plot(waveform.numpy(), alpha=0.5, label="Waveform")
    ax0.set_title(f"Raw Waveform and Boundary Labels (Text: '{text}', WPM: {wpm})")
    
    # Show boundary labels on waveform (scaled for visibility)
    b_indices = np.where(boundary_labels.numpy() > 0.5)[0]
    for idx in b_indices:
        ax0.axvline(x=idx * config.HOP_LENGTH, color='red', linestyle='--', alpha=0.8, label="Boundary" if idx == b_indices[0] else "")
    ax0.legend()

    # 2. Signal Labels (Downsampled)
    ax1 = axes[1]
    # 0:BG, 1:Dit, 2:Dah, 3:Intra, 4:Inter, 5:Word
    ax1.plot(s_downsampled.numpy(), drawstyle='steps-post', color='green', label="Signal Class")
    ax1.set_title("Downsampled Signal Labels (Model Output Resolution)")
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    ax1.set_yticklabels(['BG', 'Dit', 'Dah', 'Intra', 'Inter', 'Word'])
    ax1.grid(True, alpha=0.3)

    # 3. Boundary Labels (Downsampled)
    ax2 = axes[2]
    ax2.plot(b_downsampled.numpy(), drawstyle='steps-post', color='red', label="Boundary")
    ax2.set_title("Downsampled Boundary Labels (Model Output Resolution)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Overlay signal class on boundary plot for context
    ax2_twin = ax2.twinx()
    ax2_twin.plot(s_downsampled.numpy(), alpha=0.2, color='green', drawstyle='steps-post')
    ax2_twin.set_ylabel("Signal Class (shadow)")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved boundary visualization to {output_path}")

if __name__ == "__main__":
    main()
