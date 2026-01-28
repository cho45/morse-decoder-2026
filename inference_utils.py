import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import config

# Prosign mapping for single-token evaluation
PROSIGN_MAPPING = {ps: chr(i + 1) for i, ps in enumerate(config.PROSIGNS)}
INV_PROSIGN_MAPPING = {v: k for k, v in PROSIGN_MAPPING.items()}

def map_prosigns(text: str) -> str:
    """Replace prosign strings with single control codes for fair CER calculation."""
    sorted_prosigns = sorted(config.PROSIGNS, key=len, reverse=True)
    mapped_text = text
    for ps in sorted_prosigns:
        mapped_text = mapped_text.replace(ps, PROSIGN_MAPPING[ps])
    return mapped_text

def unmap_prosigns(text: str) -> str:
    """Replace control codes back with original prosign strings."""
    unmapped_text = ""
    for char in text:
        if char in INV_PROSIGN_MAPPING:
            unmapped_text += INV_PROSIGN_MAPPING[char]
        else:
            unmapped_text += char
    return unmapped_text

def levenshtein_prosign(a: str, b: str) -> Tuple[int, List[Tuple[str, str, str]]]:
    """
    Calculates the Levenshtein distance and backtrace between a and b with prosign handling.
    Returns:
        distance: int
        ops: List[Tuple[str, str, str]] - list of (op, ref_char, hyp_char)
    """
    a_mapped = map_prosigns(a)
    b_mapped = map_prosigns(b)

    n, m = len(a_mapped), len(b_mapped)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a_mapped[i - 1] == b_mapped[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,    # deletion
                               dp[i][j - 1] + 1,    # insertion
                               dp[i - 1][j - 1] + 1) # substitution

    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a_mapped[i - 1] == b_mapped[j - 1]:
            ref_char = INV_PROSIGN_MAPPING.get(a_mapped[i - 1], a_mapped[i - 1])
            hyp_char = INV_PROSIGN_MAPPING.get(b_mapped[j - 1], b_mapped[j - 1])
            ops.append(('match', ref_char, hyp_char))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ref_char = INV_PROSIGN_MAPPING.get(a_mapped[i - 1], a_mapped[i - 1])
            hyp_char = INV_PROSIGN_MAPPING.get(b_mapped[j - 1], b_mapped[j - 1])
            ops.append(('sub', ref_char, hyp_char))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ref_char = INV_PROSIGN_MAPPING.get(a_mapped[i - 1], a_mapped[i - 1])
            ops.append(('del', ref_char, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            hyp_char = INV_PROSIGN_MAPPING.get(b_mapped[j - 1], b_mapped[j - 1])
            ops.append(('ins', None, hyp_char))
            j -= 1
    
    return dp[n][m], ops[::-1]

def calculate_cer(ref: str, hyp: str) -> float:
    """Calculate Character Error Rate using Levenshtein distance with prosign handling."""
    ref_no_space = ref.replace(" ", "")
    hyp_no_space = hyp.replace(" ", "")
    
    if not ref_no_space:
        return 1.0 if hyp_no_space else 0.0
    
    dist, _ = levenshtein_prosign(ref_no_space, hyp_no_space)
    return dist / len(map_prosigns(ref_no_space))

def preprocess_waveform(waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Standardized preprocessing for both training and inference.
    Includes lookahead padding, spectrogram, bin cropping, and log scaling.
    """
    with torch.no_grad():
        # Ensure waveform is on correct device
        x = waveform.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Physical Lookahead Padding
        lookahead_samples = config.LOOKAHEAD_FRAMES * config.HOP_LENGTH
        x = F.pad(x, (0, lookahead_samples))
        
        # Spectrogram
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0,
            center=False
        ).to(device)
        spec = spec_transform(x)
        
        # Crop frequency bins
        f_bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
        f_bin_end = f_bin_start + config.N_BINS
        spec = spec[:, f_bin_start:f_bin_end, :]
        
        # PCEN is handled inside the model for the current architecture.
        # We pass the raw power spectrogram.
        return spec.transpose(1, 2) # (Batch, T, N_BINS)

def decode_multi_task(
    ctc_logits: torch.Tensor, 
    sig_logits: torch.Tensor, 
    bound_probs: torch.Tensor,
    id_to_char: Dict[int, str] = config.ID_TO_CHAR,
    bound_threshold: float = 0.2,
    space_threshold: float = 0.1
) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Greedy decoding with Signal Head gating and Boundary Head gating.
    Returns:
        decoded_text: str
        timed_output: List of (char, frame_index)
    """
    # Get predictions
    preds = ctc_logits.argmax(dim=-1) # (T,)
    sig_preds = sig_logits.argmax(dim=-1) # (T,)
    
    decoded_indices = []
    decoded_positions = []
    prev = -1
    
    # 1. CTC Greedy Decoding
    for t in range(len(preds)):
        idx = preds[t].item()
        if idx != 0 and idx != prev:
            decoded_indices.append(idx)
            decoded_positions.append(t)
        prev = idx
        
    # 2. Gated Space Reconstruction
    result = []
    timed_output = []

    for i, (idx, pos) in enumerate(zip(decoded_indices, decoded_positions)):
        # Append character first
        char = id_to_char.get(idx, "")
        result.append(char)
        timed_output.append((char, pos))

        # Check for inter-word space AFTER current character
        # Space is detected between current position and next character position
        # This reflects the CW signal timing: character fires → space detected → next character fires
        if i + 1 < len(decoded_positions):
            next_pos = decoded_positions[i + 1]
            # Check if word space (class 3) exists between current and next character
            # Gated by Boundary Head probability to prevent false positives in noise
            if any(sig_preds[pos:next_pos] == 3) and any(bound_probs[pos:next_pos] > space_threshold):
                result.append(" ")

    return "".join(result).strip(), timed_output

def visualize_inference(
    mels: torch.Tensor,
    ctc_logits: torch.Tensor,
    sig_logits: torch.Tensor,
    bound_probs: torch.Tensor,
    target_sig: Optional[torch.Tensor] = None,
    target_bound: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Comprehensive visualization of model predictions vs targets.
    """
    mels = mels[0].cpu().numpy() # (T, F)
    ctc_probs = torch.softmax(ctc_logits[0], dim=-1).cpu().numpy()
    sig_preds = torch.softmax(sig_logits[0], dim=-1).cpu().numpy()
    bound_probs = bound_probs[0].cpu().numpy().squeeze(-1)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # 1. Spectrogram
    axes[0].imshow(mels.T, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Input Spectrogram")
    
    # 2. CTC Spikes
    char_probs = 1.0 - ctc_probs[:, 0]
    axes[1].plot(char_probs, label="Char Prob (Sum)", color='blue')
    axes[1].set_title("CTC Character Probabilities")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Signal Head
    # 0:bg, 1:dit, 2:dah, 3:word
    colors = ['gray', 'green', 'red', 'orange']
    labels = ['Background', 'Dit', 'Dah', 'Word Space']
    for i in range(1, 4):
        axes[2].plot(sig_preds[:, i], label=labels[i], color=colors[i], alpha=0.7)
    
    if target_sig is not None:
        t_sig = target_sig.cpu().numpy()
        # Overlay target signal as semi-transparent background
        for i in range(1, 4):
            axes[2].fill_between(range(len(t_sig)), 0, (t_sig == i).astype(float) * 0.3, color=colors[i], alpha=0.2)
            
    axes[2].set_title("Signal Head Predictions")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc='upper right', fontsize='small')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Boundary Head
    axes[3].plot(bound_probs, label="Boundary Prob", color='purple')
    if target_bound is not None:
        t_bound = target_bound.cpu().numpy()
        axes[3].fill_between(range(len(t_bound)), 0, t_bound * 0.5, color='purple', alpha=0.2, label="Target Boundary")
    
    axes[3].set_title("Boundary Head Predictions")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc='upper right', fontsize='small')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
