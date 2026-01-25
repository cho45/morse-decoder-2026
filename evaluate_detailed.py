import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import argparse
import os
from typing import List, Tuple, Dict

from model import StreamingConformer
from data_gen import generate_sample, MORSE_DICT
import config

# Import decoding utilities from train.py
def decode_ctc_output(logits: np.ndarray, signal_logits: np.ndarray) -> str:
    """Decode CTC output using greedy decoding with space reconstruction.

    Args:
        logits: CTC logits (T, C)
        signal_logits: Signal class logits (T, NUM_SIGNAL_CLASSES)

    Returns:
        Decoded text string
    """
    preds = logits.argmax(axis=-1)  # (T,)
    sig_preds = signal_logits.argmax(axis=-1)  # (T,)

    # CTC greedy decoding
    decoded_indices = []
    decoded_positions = []
    prev = -1
    for t in range(len(preds)):
        idx = preds[t]
        if idx != 0 and idx != prev:  # Skip blank(0) and repeats
            decoded_indices.append(idx)
            decoded_positions.append(t)
        prev = idx

    # Space reconstruction using signal predictions
    result = []
    last_pos = 0
    for idx, pos in zip(decoded_indices, decoded_positions):
        # Check for inter-word space (class 5) between last position and current
        if any(sig_preds[last_pos:pos] == 5):
            result.append(" ")
        result.append(config.ID_TO_CHAR.get(idx, "?"))
        last_pos = pos

    return "".join(result).strip()

def get_args():
    parser = argparse.ArgumentParser(description="Detailed evaluation of CW Decoder with SNR sweep")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default="CQ DE KILO CODE K", help="Text to test")
    parser.add_argument("--wpm", type=int, default=25, help="Words per minute")
    parser.add_argument("--snrs", type=int, nargs="+", default=[30, 20, 10, 5, 0], help="SNR values to test")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save plots and wavs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-mode", type=str, default="streaming", choices=["streaming", "batch"],
                        help="Inference mode: streaming (chunk-by-chunk) or batch (full sequence)")
    parser.add_argument("--chunk-size", type=int, default=16, help="Chunk size for streaming inference")
    return parser.parse_args()

class DetailedEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = StreamingConformer(
            n_mels=config.N_MELS,
            num_classes=config.NUM_CLASSES,
            d_model=config.D_MODEL,
            n_head=config.N_HEAD,
            num_layers=config.NUM_LAYERS,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=500.0,
            f_max=900.0,
            center=False
        ).to(self.device)

    def run_inference(self, waveform: torch.Tensor, mode: str = "streaming", chunk_size: int = 16) -> Dict[str, np.ndarray]:
        """Runs inference and collects all internal states.

        Args:
            waveform: Input audio waveform
            mode: 'streaming' for chunk-by-chunk inference, 'batch' for full sequence inference
            chunk_size: Chunk size for streaming mode (must be multiple of SUBSAMPLING_RATE)
        """
        # Preprocess full waveform to mel
        with torch.no_grad():
            x = waveform.to(self.device).unsqueeze(0)
            mels = self.mel_transform(x)
            # Match train.py scaling: log1p(mel * 100) / 5.0
            mels = torch.log1p(mels * 100.0) / 5.0
            mels = mels.transpose(1, 2) # (1, T_mel, F)

            if mode == "batch":
                # Batch inference (same as train.py validation)
                (ctc, sig, bound), _ = self.model(mels)
                return {
                    "mels": mels.cpu().numpy()[0],
                    "ctc_logits": ctc.cpu().numpy()[0],
                    "sig_logits": sig.cpu().numpy()[0],
                    "bound_logits": bound.cpu().numpy()[0]
                }
            else:
                # Streaming inference loop
                states = None
                all_ctc_logits = []
                all_sig_logits = []
                all_bound_logits = []

                for i in range(0, mels.size(1), chunk_size):
                    chunk = mels[:, i:i+chunk_size, :]
                    if chunk.size(1) < chunk_size:
                        # Pad last chunk if necessary to satisfy subsampling requirements
                        # However, StreamingConformer.subsampling handles small chunks by returning empty
                        # if they are too small. To be safe, we can skip or pad.
                        if chunk.size(1) % config.SUBSAMPLING_RATE != 0:
                            continue

                    (ctc, sig, bound), states = self.model(chunk, states)
                    all_ctc_logits.append(ctc.cpu().numpy())
                    all_sig_logits.append(sig.cpu().numpy())
                    all_bound_logits.append(bound.cpu().numpy())

                return {
                    "mels": mels.cpu().numpy()[0],
                    "ctc_logits": np.concatenate(all_ctc_logits, axis=1)[0],
                    "sig_logits": np.concatenate(all_sig_logits, axis=1)[0],
                    "bound_logits": np.concatenate(all_bound_logits, axis=1)[0]
                }

    def visualize(self, data: Dict[str, np.ndarray],
                  gt_sig: np.ndarray, gt_bound: np.ndarray,
                  text: str, snr: float, output_path: str, mode: str = "streaming"):
        """Creates a detailed plot of the model's internal states."""
        mels = data["mels"].T # (F, T_mel)
        ctc_probs = torch.softmax(torch.from_numpy(data["ctc_logits"]), dim=-1).numpy()
        sig_probs = torch.softmax(torch.from_numpy(data["sig_logits"]), dim=-1).numpy()
        bound_probs = torch.sigmoid(torch.from_numpy(data["bound_logits"])).numpy()[:, 0]

        # Decode CTC output
        decoded_text = decode_ctc_output(data["ctc_logits"], data["sig_logits"])

        # Time axes
        t_mel = np.arange(mels.shape[1]) * config.HOP_LENGTH / config.SAMPLE_RATE
        t_model = np.arange(ctc_probs.shape[0]) * (config.HOP_LENGTH * config.SUBSAMPLING_RATE) / config.SAMPLE_RATE

        fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 2, 1, 1]})

        # 1. Mel Spectrogram
        axes[0].imshow(mels, aspect='auto', origin='lower', extent=[t_mel[0], t_mel[-1], 0, config.N_MELS])
        axes[0].set_ylabel("Mel Bin")
        title_text = f"Detailed Analysis: '{text}' at SNR={snr}dB ({mode} inference)\nDecoded: '{decoded_text}'"
        axes[0].set_title(title_text)

        # 2. CTC Logits (All decoded classes)
        # Get all classes that appear in the decoded output
        preds = data["ctc_logits"].argmax(axis=-1)
        decoded_class_ids = set()
        prev = -1
        for t in range(len(preds)):
            idx = preds[t]
            if idx != 0 and idx != prev:
                decoded_class_ids.add(idx)
            prev = idx

        # Plot all decoded classes
        decoded_class_ids = sorted(decoded_class_ids)
        for idx in decoded_class_ids:
            axes[1].plot(t_model, ctc_probs[:, idx], label=config.ID_TO_CHAR.get(idx, f"ID:{idx}"), linewidth=1.5)
        axes[1].plot(t_model, ctc_probs[:, 0], label="blank", color='gray', alpha=0.3, linestyle='--')
        axes[1].set_ylabel("CTC Prob")
        axes[1].legend(loc='upper right', fontsize='small', ncol=2)
        axes[1].set_ylim(0, 1.1)

        # 3. Signal Head
        # Classes: 1: Dit, 2: Dah, 3: Intra-char space, 4: Inter-char space, 5: Inter-word space
        sig_labels = ["None", "Dit", "Dah", "Intra", "Inter-Char", "Inter-Word"]
        colors = ['#eeeeee', '#ff9999', '#99ff99', '#9999ff', '#ffff99', '#ff99ff']
        
        # Ground Truth Background
        gt_t_mel = np.arange(len(gt_sig)) * config.HOP_LENGTH / config.SAMPLE_RATE
        for i in range(1, 6):
            mask = (gt_sig == i)
            axes[2].fill_between(gt_t_mel, 0, 1, where=mask, color=colors[i], alpha=0.3, label=f"GT {sig_labels[i]}")
        
        # Predictions
        for i in range(1, 6):
            axes[2].plot(t_model, sig_probs[:, i], label=f"Pred {sig_labels[i]}", color=colors[i], linewidth=2)
        
        axes[2].set_ylabel("Signal Prob")
        axes[2].set_ylim(0, 1.1)
        # Unique legends
        handles, labels = axes[2].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[2].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='x-small', ncol=2)

        # 4. Boundary Head
        axes[3].plot(t_model, bound_probs, label="Boundary Pred", color='blue')
        # GT Boundaries as vertical lines
        gt_bound_indices = np.where(gt_bound > 0.5)[0]
        for idx in gt_bound_indices:
            axes[3].axvline(x=idx * config.HOP_LENGTH / config.SAMPLE_RATE, color='red', alpha=0.5, linestyle=':', label="GT Boundary" if idx == gt_bound_indices[0] else "")
        axes[3].set_ylabel("Bound Prob")
        axes[3].set_ylim(0, 1.1)
        axes[3].legend(loc='upper right', fontsize='small')

        # 5. Audio Waveform (Downsampled for display)
        # waveform is too large, so we just show its envelope or skip
        # Let's show the ground truth signal class as a color bar
        axes[4].imshow(gt_sig[None, :], aspect='auto', cmap='tab10', extent=[gt_t_mel[0], gt_t_mel[-1], 0, 1], alpha=0.8)
        axes[4].set_yticks([])
        axes[4].set_ylabel("GT Classes")
        axes[4].set_xlabel("Time (sec)")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = DetailedEvaluator(args.checkpoint, device=args.device)

    print(f"Starting detailed evaluation for text: '{args.text}'")
    print(f"SNRs: {args.snrs}")
    print(f"Inference mode: {args.inference_mode}")
    if args.inference_mode == "streaming":
        print(f"Chunk size: {args.chunk_size}")

    for snr in args.snrs:
        print(f"Processing SNR: {snr}dB...")

        # Generate sample
        waveform, text, gt_sig, gt_bound = generate_sample(
            args.text, wpm=args.wpm, snr_db=snr
        )

        # Save WAV
        wav_path = os.path.join(args.output_dir, f"sample_snr{snr}_{args.inference_mode}.wav")
        wf_np = waveform.numpy()
        scipy.io.wavfile.write(wav_path, config.SAMPLE_RATE, (wf_np * 32767).astype(np.int16))

        # Run inference
        results = evaluator.run_inference(waveform, mode=args.inference_mode, chunk_size=args.chunk_size)

        # Decode and display results
        decoded_text = decode_ctc_output(results["ctc_logits"], results["sig_logits"])
        print(f"  Ground Truth: {args.text}")
        print(f"  Decoded:      {decoded_text}")

        # Check space detection accuracy
        sig_preds = results["sig_logits"].argmax(axis=-1)
        num_inter_word_spaces = (sig_preds == 5).sum()
        print(f"  Inter-word spaces detected (class 5): {num_inter_word_spaces}")

        # Show signal class distribution
        sig_probs = np.exp(results["sig_logits"] - np.max(results["sig_logits"], axis=-1, keepdims=True))
        sig_probs = sig_probs / sig_probs.sum(axis=-1, keepdims=True)
        for cls in range(6):
            cls_name = ["Bg", "Dit", "Dah", "Intra", "Inter-char", "Inter-word"][cls]
            max_prob = sig_probs[:, cls].max()
            mean_prob = sig_probs[:, cls].mean()
            print(f"    Class {cls} ({cls_name}): max={max_prob:.4f}, mean={mean_prob:.4f}")

        # Visualize
        plot_path = os.path.join(args.output_dir, f"analysis_snr{snr}_{args.inference_mode}.png")
        evaluator.visualize(
            results, gt_sig.numpy(), gt_bound.numpy(),
            args.text, snr, plot_path, mode=args.inference_mode
        )

    print(f"Evaluation complete. Results saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()