import torch
import torchaudio
import numpy as np
import config
from data_gen import generate_sample

def analyze_snr(snr_db):
    print(f"\n--- Analyzing SNR: {snr_db}dB ---")
    # Generate sample
    waveform, text, gt_sig, gt_bound = generate_sample(
        "CQ", wpm=25, snr_db=snr_db
    )
    
    # Preprocessing (from evaluate_detailed.py / train.py)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=500.0,
        f_max=900.0,
        center=False
    )
    
    with torch.no_grad():
        mels = mel_transform(waveform.unsqueeze(0))
        # Raw Mel stats
        print(f"Raw Mel - Min: {mels.min():.6f}, Max: {mels.max():.6f}, Mean: {mels.mean():.6f}")
        
        # Scaling: log1p(mel * 100) / 5.0
        mels_scaled = torch.log1p(mels * 100.0) / 5.0
        print(f"Scaled Mel - Min: {mels_scaled.min():.6f}, Max: {mels_scaled.max():.6f}, Mean: {mels_scaled.mean():.6f}")
        
        # Signal vs Noise stats
        # gt_sig: 1: Dit, 2: Dah, 0: Background
        # We need to align gt_sig with mel frames
        num_frames = mels_scaled.size(2)
        sig_mask = (gt_sig[:num_frames] > 0)
        noise_mask = (gt_sig[:num_frames] == 0)
        
        if sig_mask.any():
            sig_mel = mels_scaled[0, :, :len(sig_mask)][:, sig_mask]
            print(f"Signal frames - Mean: {sig_mel.mean():.6f}, Max: {sig_mel.max():.6f}")
        if noise_mask.any():
            noise_mel = mels_scaled[0, :, :len(noise_mask)][:, noise_mask]
            print(f"Noise frames - Mean: {noise_mel.mean():.6f}, Max: {noise_mel.max():.6f}")

if __name__ == "__main__":
    for snr in [30, 10, 0, -5]:
        analyze_snr(snr)