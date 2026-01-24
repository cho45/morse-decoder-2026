import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from data_gen import CWDataset, generate_sample
import config
import train

def main():
    parser = argparse.ArgumentParser(description="Visualize training data for CW Decoder")
    parser.add_argument("--phase", type=int, default=22, help="Curriculum phase (1-22)")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default="diagnostics/data_visualization.png", help="Output path for the plot")
    args = parser.parse_args()

    # Setup Directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize Dataset
    dataset = CWDataset(num_samples=args.num_samples)
    
    # Mock Trainer to use its preprocessing logic
    class MockArgs:
        def __init__(self):
            self.lr = 3e-4
            self.freeze_encoder = False
            self.samples_per_epoch = 1000
            self.batch_size = 1
            self.curriculum_phase = args.phase
    
    trainer = train.Trainer(MockArgs())
    
    # Apply curriculum settings to dataset (similar to train_epoch)
    epoch = args.phase * 2 # Approximate epoch for the phase
    # We manually trigger the curriculum setting logic by calling a simplified version
    # Since we want to see what the trainer would set:
    if args.phase <= 19:
        KOCH_CHARS = "KMRSTLOPANWGDUXZQVFHYCJB7512908346/?"
        num_chars = min(2 + (args.phase - 1) * 2, len(KOCH_CHARS))
        current_chars = KOCH_CHARS[:num_chars]
        dataset.min_wpm = 20
        dataset.max_wpm = 20
        dataset.min_snr = 100.0
        dataset.max_snr = 100.0
        dataset.jitter_max = 0.0
        dataset.weight_var = 0.0
        dataset.chars = current_chars
        dataset.min_len = 2
        dataset.max_len = 6
        phase_name = f"Koch Phase {args.phase}"
    elif args.phase == 20:
        dataset.min_snr = 50.0
        dataset.max_snr = 60.0
        phase_name = "Phase 20: Full Chars, Clean"
    elif args.phase == 21:
        dataset.min_wpm = 18
        dataset.max_wpm = 25
        dataset.min_snr = 30.0
        dataset.max_snr = 45.0
        dataset.jitter_max = 0.03
        dataset.weight_var = 0.05
        phase_name = "Phase 21: Slight Variations"
    else:
        dataset.min_wpm = 10
        dataset.max_wpm = 45
        dataset.min_snr = 5.0
        dataset.max_snr = 30.0
        dataset.jitter_max = 0.10
        dataset.weight_var = 0.15
        phase_name = "Phase 22: Realistic"

    print(f"Visualizing {args.num_samples} samples from {phase_name}")

    fig, axes = plt.subplots(args.num_samples, 2, figsize=(15, 4 * args.num_samples))
    if args.num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(args.num_samples):
        waveform, label, wpm = dataset[i]
        
        # Preprocess using trainer's logic
        # waveform is (T,) -> needs (1, T) for mel_transform
        wf_batch = waveform.unsqueeze(0)
        mels, _ = trainer.compute_mels_and_lengths(wf_batch, torch.tensor([waveform.shape[0]]))
        # mels: (1, T_frames, N_MELS)
        
        # Plot Waveform
        ax_wf = axes[i, 0]
        ax_wf.plot(waveform.numpy())
        ax_wf.set_title(f"Sample {i+1}: '{label}' (WPM: {wpm})\nWaveform")
        ax_wf.set_xlabel("Samples")
        ax_wf.set_ylabel("Amplitude")
        ax_wf.grid(True, alpha=0.3)

        # Plot Mel Spectrogram
        ax_mel = axes[i, 1]
        mel_data = mels[0].transpose(0, 1).cpu().numpy() # (N_MELS, T_frames)
        im = ax_mel.imshow(mel_data, aspect='auto', origin='lower', interpolation='nearest')
        ax_mel.set_title(f"Mel Spectrogram (N_MELS={config.N_MELS})")
        ax_mel.set_xlabel("Frames")
        ax_mel.set_ylabel("Mel Bin")
        fig.colorbar(im, ax=ax_mel)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()