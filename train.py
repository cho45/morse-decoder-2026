import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
import string
import csv
from typing import List, Tuple

from data_gen import CWDataset, MORSE_DICT, generate_sample
from model import StreamingConformer
import config
import visualize_logs

# Use centralized config
CHARS = config.CHARS
CHAR_TO_ID = config.CHAR_TO_ID
ID_TO_CHAR = config.ID_TO_CHAR
NUM_CLASSES = config.NUM_CLASSES

# Standard Koch order: K, M, R, S, T, L, O, P, I, A, N, W, G, D, U, X, Z, Q, V, F, Y, C, H, J, B, L, 7, 5, 1, 2, 9, 0, 8, 3, 4, 6
# (Simplified and adjusted to match our vocabulary)
KOCH_CHARS = "KMRSTLOPANWGDUXZQVFHYCJB7512908346/?"

# 誤認識しやすい文字のペア。新しい文字（key）を導入する際、
# 既に学習済みの文字（value）も同時に重点サンプリングに含めることで、
# モデルにその差異を強制的に学習させる（対照学習）。
# 診断ツール（evaluate_confusion.py）の結果に基づき、エビデンスベースで更新。
# 長点過多（Oへの吸収）や末尾リズムの誤認（W vs P/J）を重点的に矯正する。
CONFUSION_PAIRS = {
    'L': 'R',    # .-.. vs .-.
    'F': 'L',    # ..-. vs .-..
    'W': 'PAGJ', # .-- (W) vs .--. (P), .- (A), --. (G), .--- (J)
    'P': 'WJ',   # .--. vs .-- (W), .--- (J)
    'B': 'D',    # -... vs -..
    'X': 'P',    # -..- vs .--.
    'Y': 'Q',    # -.-- vs --.-
    'U': 'V',    # ..- vs ...-
    '8': 'O',    # ---.. vs --- (O)
    '0': 'O',    # ----- vs --- (O)
    '7': 'Z',    # --... vs --.. (Z)
    '4': 'X',    # ....- vs -..- (X)
    'I': 'T',    # .. vs - (T)
    'A': 'W',    # .- vs .--
    'M': 'O',    # -- vs ---
    '<AR>': 'C', # .-.-. vs -.-. (C)
    '<BT>': 'X', # -...- vs -..- (X)
}

# Mapping Prosigns to ASCII control codes for single-token Levenshtein calculation
# Using \x01, \x02, ... for prosigns to ensure they are treated as a single character.
PROSIGN_MAPPING = {ps: chr(i + 1) for i, ps in enumerate(config.PROSIGNS)}
INV_PROSIGN_MAPPING = {v: k for k, v in PROSIGN_MAPPING.items()}

def map_prosigns(text: str) -> str:
    """Replace prosign strings with their corresponding control codes."""
    # Order by length descending to avoid partial matches (e.g., <BT> vs <BT)
    # although our prosigns are quite distinct.
    sorted_prosigns = sorted(config.PROSIGNS, key=len, reverse=True)
    mapped_text = text
    for ps in sorted_prosigns:
        mapped_text = mapped_text.replace(ps, PROSIGN_MAPPING[ps])
    return mapped_text

def unmap_prosigns(text: str) -> str:
    """Replace control codes back with their original prosign strings."""
    unmapped_text = ""
    for char in text:
        if char in INV_PROSIGN_MAPPING:
            unmapped_text += INV_PROSIGN_MAPPING[char]
        else:
            unmapped_text += char
    return unmapped_text

def levenshtein(a, b):
    """
    Calculates the Levenshtein distance and backtrace between a and b.
    Returns:
        distance: int
        ops: List[Tuple[str, str, str]] - list of (op, ref_char, hyp_char)
             op is one of 'match', 'sub', 'ins', 'del'
    """
    # Pre-process strings to map prosigns to single control codes
    a = map_prosigns(a)
    b = map_prosigns(b)

    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,    # deletion
                               dp[i][j - 1] + 1,    # insertion
                               dp[i - 1][j - 1] + 1) # substitution

    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ref_char = INV_PROSIGN_MAPPING.get(a[i - 1], a[i - 1])
            hyp_char = INV_PROSIGN_MAPPING.get(b[j - 1], b[j - 1])
            ops.append(('match', ref_char, hyp_char))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ref_char = INV_PROSIGN_MAPPING.get(a[i - 1], a[i - 1])
            hyp_char = INV_PROSIGN_MAPPING.get(b[j - 1], b[j - 1])
            ops.append(('sub', ref_char, hyp_char))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ref_char = INV_PROSIGN_MAPPING.get(a[i - 1], a[i - 1])
            ops.append(('del', ref_char, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            hyp_char = INV_PROSIGN_MAPPING.get(b[j - 1], b[j - 1])
            ops.append(('ins', None, hyp_char))
            j -= 1
    
    return dp[n][m], ops[::-1]

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute Mel filterbank using config
        # Single-bin MelSpectrogram focusing on 700Hz to maximize information density
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=500.0,
            f_max=900.0,
            center=False # Use False for strict streaming causality
        ).to(self.device)
        
        self.model = StreamingConformer(
            n_mels=config.N_MELS,
            num_classes=NUM_CLASSES,
            d_model=getattr(args, 'd_model', config.D_MODEL),
            n_head=getattr(args, 'n_head', config.N_HEAD),
            num_layers=getattr(args, 'num_layers', config.NUM_LAYERS),
            dropout=getattr(args, 'dropout', config.DROPOUT)
        ).to(self.device)

        if args.freeze_encoder:
            # Note: 停滞打破のため、全層解放モードを優先することを検討してください。
            print("Freezing encoder parameters (except heads)...")
            for name, param in self.model.named_parameters():
                # ctc_head, signal_head, boundary_head 以外を固定
                if "ctc_head" not in name and "signal_head" not in name and "boundary_head" not in name:
                    param.requires_grad = False
        
        # 余計なハイパーパラメータ引数への依存を減らし、標準的な AdamW 設定を使用
        # 学習率は warmup 後に args.lr に到達するように調整
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=args.lr, 
            weight_decay=getattr(args, 'weight_decay', 1e-5)
        )
        
        # 指標を CER に変更し、忍耐強めに設定 (マルチタスク学習の変動を考慮して 10 に強化)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        # CTC Loss: zero_infinity=True to handle edge cases in early training
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        # Multi-class Cross Entropy for Signal Detection (Multi-task)
        # 0: Background, 1: Dit, 2: Dah, 3: Intra-char space, 4: Inter-char space, 5: Inter-word space
        self.signal_criterion = nn.CrossEntropyLoss(reduction='none')
        # Binary Cross Entropy for Character Boundary Detection
        # Positive weight to handle class imbalance (boundaries are rare)
        pos_weight = torch.tensor([20.0]).to(self.device)
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        
        # Dataset
        self.train_dataset = CWDataset(num_samples=args.samples_per_epoch)
        self.val_dataset = CWDataset(num_samples=args.samples_per_epoch // 10)
        
        self.history_path = os.path.join(args.save_dir, "history.csv")
        
        # Adaptive Curriculum State
        self.current_phase = 1
        self.phases_since_last_advance = 0
        self.min_epochs_per_phase = 3
        self.cer_threshold_to_advance = 0.01

        # Initialize phase from args if provided
        if self.args.curriculum_phase > 0:
            self.current_phase = self.args.curriculum_phase

    def update_curriculum(self, val_cer):
        """
        Update curriculum phase based on validation CER.
        Only active if args.curriculum_phase is 0 (auto mode).
        """
        # Always increment counter
        self.phases_since_last_advance += 1
        
        # Only update phase if auto mode is enabled
        if self.args.curriculum_phase == 0:
            # Logic: If CER is low enough AND we've spent enough time in this phase
            if val_cer < self.cer_threshold_to_advance and self.phases_since_last_advance >= self.min_epochs_per_phase:
                # Calculate max phase dynamically based on KOCH_CHARS length
                # Phase 1: 2 chars
                # Phase 2: 4 chars
                # Phase N: len(KOCH_CHARS) chars -> N = len(KOCH_CHARS) - 2
                # + 2 phases for Full Clean and Slight Variations
                # + 1 phase for Realistic
                # 7 phases were: FullClean, SlightVar, Practical, Boundary, NegEntry, DeepNeg, TrueExtreme
                # Now we expand to more steps for smoother SNR transition and fading resistance
                max_phase = (len(KOCH_CHARS) - 4) + 2 + 14
                
                if self.current_phase < max_phase:
                    print(f"*** PERFORMANCE GOOD (CER {val_cer:.4f} < {self.cer_threshold_to_advance}). ADVANCING TO PHASE {self.current_phase + 1} ***")
                    self.current_phase += 1
                    self.phases_since_last_advance = 0
                    
                    # Reset optimizer LR to boost learning for new phase
                    # 難易度の高い文字（Lなど）が導入される場合は、少し低めのLRから開始して慎重に学習させる
                    new_char = ""
                    num_chars = 0
                    if self.current_phase == 1: num_chars = 2
                    elif self.current_phase == 2: num_chars = 4
                    else: num_chars = 4 + (self.current_phase - 2)
                    
                    if num_chars <= len(KOCH_CHARS):
                        new_char = KOCH_CHARS[num_chars-1]
                    
                    reset_lr = self.args.lr
                    if new_char in CONFUSION_PAIRS:
                        reset_lr = self.args.lr * 0.3 # 難易度が高い場合は LR を抑える
                        print(f"[Curriculum] Difficult char '{new_char}' detected. Using cautious LR: {reset_lr}")
                    
                    print(f"[Curriculum] Resetting Learning Rate to {reset_lr}")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = reset_lr
                else:
                    print(f"*** MAX PHASE REACHED. CONTINUING REFINEMENT ***")
            else:
                print(f"[DEBUG] Phase {self.current_phase}: CER {val_cer:.4f} (Target < {self.cer_threshold_to_advance}), Epochs {self.phases_since_last_advance}/{self.min_epochs_per_phase} - WAITING")
                
                # 停滞打破のための LR 再加熱 (Re-heating)
                # 10エポック停滞したら、LR を初期値の半分まで戻して局所解からの脱出を試みる
                # 以降、5エポックごとに再加熱を繰り返す
                if self.phases_since_last_advance >= 10 and (self.phases_since_last_advance - 10) % 5 == 0:
                    # 強力な再加熱: 停滞時は初期 LR まで戻す
                    reheat_lr = self.args.lr
                    print(f"*** STAGNATION DETECTED ({self.phases_since_last_advance} epochs). RE-HEATING LR TO {reheat_lr} ***")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = reheat_lr

    def load_checkpoint_state(self, checkpoint):
        """Load curriculum state from checkpoint."""
        if self.args.curriculum_phase == 0 and 'curriculum_phase' in checkpoint:
            self.current_phase = checkpoint['curriculum_phase']
        
        if 'phases_since_last_advance' in checkpoint:
            self.phases_since_last_advance = checkpoint['phases_since_last_advance']
            
    def collate_fn(self, batch: List[Tuple[torch.Tensor, str, int, torch.Tensor, torch.Tensor]]):
        # batch is a list of (waveform, text, wpm, signal_labels, boundary_labels)
        waveforms, texts, wpms, signal_labels, boundary_labels = zip(*batch)
        
        # Physical Lookahead Alignment:
        # Add LOOKAHEAD_FRAMES at the end of each waveform to allow the causal model
        # to "look ahead" into the future signal for the current label.
        lookahead_samples = config.LOOKAHEAD_FRAMES * config.HOP_LENGTH
        
        # Pad waveforms
        lengths = torch.tensor([w.shape[0] for w in waveforms])
        padded_waveforms = torch.zeros(len(waveforms), lengths.max() + lookahead_samples)
        for i, w in enumerate(waveforms):
            # Place signal at the beginning
            padded_waveforms[i, :w.shape[0]] = w
            
        # Pad labels
        # signal_labels are already in mel frames units.
        # We need to calculate the target number of frames after subsampling.
        l_total = padded_waveforms.size(1)
        l_mel = (l_total - config.N_FFT) // config.HOP_LENGTH + 1
        padding_t = 2 # From compute_mels_and_lengths
        input_lengths_val = (l_mel + padding_t - 3) // 2 + 1
        
        padded_signals = torch.zeros(len(signal_labels), input_lengths_val, dtype=torch.long)
        padded_boundaries = torch.zeros(len(boundary_labels), input_lengths_val, dtype=torch.float32)
        for i, (s, b) in enumerate(zip(signal_labels, boundary_labels)):
            # Downsample labels to match model output frames (Current subsampling is 2x)
            s_downsampled = s[::config.SUBSAMPLING_RATE]
            b_downsampled = b[::config.SUBSAMPLING_RATE]
            length = min(len(s_downsampled), input_lengths_val)
            padded_signals[i, :length] = s_downsampled[:length].long()
            padded_boundaries[i, :length] = b_downsampled[:length].float()

        # Encode labels
        label_list = []
        label_lengths = []
        for text in texts:
            # Tokenize text to handle prosigns
            tokens = []
            i = 0
            while i < len(text):
                if text[i] == '<':
                    end = text.find('>', i)
                    if end != -1:
                        token = text[i:end+1]
                        if token in MORSE_DICT:
                            tokens.append(token)
                            i = end + 1
                            continue
                
                # Check for multi-char tokens like CQ, DE
                found = False
                for ps in config.PROSIGNS:
                    if text.startswith(ps, i):
                        tokens.append(ps)
                        i += len(ps)
                        found = True
                        break
                if not found:
                    if text[i] in CHAR_TO_ID:
                        tokens.append(text[i])
                    i += 1
            encoded = [CHAR_TO_ID[t] for t in tokens]
            label_list.extend(encoded)
            label_lengths.append(len(encoded))
            
        return padded_waveforms, torch.tensor(label_list, dtype=torch.long), lengths, torch.tensor(label_lengths, dtype=torch.long), texts, torch.tensor(wpms), padded_signals, padded_boundaries

    def compute_mels_and_lengths(self, waveforms, lengths):
        x = waveforms.to(self.device)
        mels = self.mel_transform(x) # (B, F, T)
        
        # Log scaling and robust normalization
        # Use a fixed scaling factor to avoid blowing up silence in InstanceNorm
        mels = torch.log1p(mels * 100.0)
        
        # Simple global-style normalization: scale to roughly [0, 1] range
        # Based on typical log1p(mel * 100) values where max is around 5-10
        mels = mels / 5.0
        
        mels = mels.transpose(1, 2) # (B, T, F)
        
        # 物理法則に基づいた厳密なフレーム計算 (torchaudio center=False の仕様)
        # Mel frames: floor((L - N_FFT) / HOP_LENGTH) + 1
        # waveforms は padding 済み (max_len + lookahead) なので、
        # input_lengths は「モデルが出力する全フレーム数」を指すようにする。
        # CTC Loss は input_lengths までの logits を使用する。
        l_total = torch.tensor(waveforms.size(1), device=self.device)
        l_mel = torch.div(l_total - config.N_FFT, config.HOP_LENGTH, rounding_mode='floor') + 1
        
        # ConvSubsampling in model.py uses kernel_size=3, stride=2, padding_t=2
        padding_t = 2
        input_lengths_val = torch.div((l_mel + padding_t) - 3, 2, rounding_mode='floor') + 1
        
        # 全バッチで同じ長さ（padded_waveforms の長さ）になるはず
        input_lengths = torch.full((waveforms.size(0),), input_lengths_val.item(), dtype=torch.long, device=self.device)

        return mels, input_lengths

    def compute_loss(self, logits, signal_logits, boundary_logits, targets, target_lengths, input_lengths, signal_targets, boundary_targets):
        """損失関数の一括計算。GPUメモリ効率を最大化した実装。"""
        input_lengths = torch.clamp(input_lengths, max=logits.size(1))
        batch_size, max_time, _ = logits.size()
        mask = torch.arange(max_time, device=self.device).expand(batch_size, max_time) < input_lengths.unsqueeze(1)
        mask = mask.float() # (B, T)

        # 1. CTC Loss (生のロジットを使用し勾配飢餓を防ぐ)
        logits_t = logits.transpose(0, 1).log_softmax(2)
        ctc_loss = self.criterion(logits_t, targets, input_lengths, target_lengths)

        # 2. Illegal Spike Penalty (数学的最適化: (B, T, C) 次元を完全に排除)
        # 全文字クラスの確率の和は 1.0 - Blank確率 と等価。
        # log_softmax の結果を再利用し、exp を取ることで巨大なテンソルの生成を回避。
        log_probs = torch.log_softmax(logits, dim=-1)
        blank_probs = torch.exp(log_probs[:, :, 0])
        char_probs_sum = 1.0 - blank_probs # 全文字クラスの確率の総和 (B, T)
        
        # 境界フラグが立っていない場所での文字確率の出現を厳しく罰する。
        # アライメント崩壊（末尾への溜め込み）を防ぐため、バウンダリ以外の全区間（ON区間含む）をペナルティ対象とする。
        # これにより、CTCスパイクをバウンダリのピンポイントな位置へ強制的に誘導する。
        bound_t = boundary_targets[:, :logits.size(1)].to(self.device)
        
        # ペナルティ対象： バウンダリでないすべての場所
        penalty_mask = (1.0 - bound_t)
        illegal_spike_loss = (char_probs_sum * penalty_mask * mask).sum() / (mask.sum() + 1e-6)

        # 3. Signal Detection Loss
        sig_t = signal_targets[:, :signal_logits.size(1)].to(self.device)
        raw_sig_loss = self.signal_criterion(signal_logits.transpose(1, 2), sig_t)
        sig_loss = (raw_sig_loss * mask).sum() / (mask.sum() + 1e-6)
        
        # 4. Boundary Detection Loss
        bound_t_bce = boundary_targets[:, :boundary_logits.size(1)].to(self.device).unsqueeze(-1)
        raw_bound_loss = self.boundary_criterion(boundary_logits, bound_t_bce).squeeze(-1)
        bound_loss = (raw_bound_loss * mask).sum() / (mask.sum() + 1e-6)

        # 損失の重みバランスを調整:
        # アライメント崩壊を矯正しつつ、Blankへの過度なバイアス（萎縮）を防ぐため重みを 2.0 に調整
        total_loss = 5.0 * ctc_loss + 1.0 * sig_loss + 5.0 * bound_loss + 2.0 * illegal_spike_loss
        
        return total_loss, {'ctc': ctc_loss, 'sig': sig_loss, 'bound': bound_loss, 'penalty': illegal_spike_loss}

    def train_epoch(self, epoch):
        self.model.train()
        # Koch Method Curriculum
        # 停滞を打破するため、文字を1つずつ導入し、かつ混乱しやすいペアを重点的に学習させる。
        
        # Use the internal current_phase
        phase = self.current_phase
        if phase == 0: phase = 1 # Safety fallback

        # Define phases explicitly for fine-grained control
        # Phase 1: 2 chars (K, M)
        # Phase 2: 4 chars (K, M, R, S) - Standard Koch start
        # Phase 3+: Add 1 char at a time
        if phase == 1:
            num_chars = 2
        elif phase == 2:
            num_chars = 4
        else:
            # Phase 3 -> 5 chars, Phase 4 -> 6 chars...
            # Formula: 4 + (phase - 2)
            num_chars = 4 + (phase - 2)
            
        max_koch_phase = len(KOCH_CHARS) - 4 + 2 # When num_chars == len(KOCH_CHARS)
        
        self.train_dataset.min_len = 5
        self.train_dataset.max_len = 6
        self.train_dataset.min_wpm = 15
        self.train_dataset.max_wpm = 40

        if num_chars <= len(KOCH_CHARS):
            # Koch Phases: Gradually increase character set
            current_chars = KOCH_CHARS[:num_chars]
            
            # Determine new chars for focus sampling
            if phase == 1:
                new_chars = current_chars
            elif phase == 2:
                new_chars = current_chars[2:] # R, S
            else:
                # Add 1 char at a time
                new_chars = current_chars[-1:]
                
                # 対照学習: 混乱しやすいペアがある場合、それも Focus Chars に含める
                # 診断ツール（evaluate_confusion.py）の結果に基づき、エビデンスベースで更新。
                for nc in list(new_chars):
                    if nc in CONFUSION_PAIRS:
                        pair_char = CONFUSION_PAIRS[nc]
                        if pair_char in current_chars:
                            print(f"  [Contrastive] Adding '{pair_char}' to focus to distinguish from '{nc}'")
                            new_chars += pair_char
            
            # カリキュラム設定の厳格な復元 (VRAM 爆発の真因: min_len/max_len の指定漏れを修正)
            # 長いシーケンスは Attention のメモリを O(T^2) で消費するため、初期学習では短く制限する。
            self.train_dataset.min_wpm = 18
            self.train_dataset.max_wpm = 22
            self.train_dataset.min_snr = 100.0 # Clean
            self.train_dataset.max_snr = 100.0
            self.train_dataset.jitter_max = 0.0
            self.train_dataset.weight_var = 0.0
            self.train_dataset.chars = current_chars
            self.train_dataset.min_len = 5
            self.train_dataset.max_len = 6
            
            # 停滞打破のため、ドリルモード（高比率 Focus）を適用
            self.train_dataset.focus_prob = 0.8 if self.phases_since_last_advance > 5 else 0.5
            
            # ドリルモード時は、新文字だけでなく既知の混同ペアも強制的に Focus に含める
            if self.phases_since_last_advance > 5:
                focus_list = list(new_chars)
                for char in current_chars:
                    if char in CONFUSION_PAIRS:
                        for pair_char in CONFUSION_PAIRS[char]:
                            if pair_char in current_chars and pair_char not in focus_list:
                                focus_list.append(pair_char)
                new_chars = "".join(focus_list)
                print(f"  [Drill Mode] Intensifying focus on: {new_chars}")

            # Apply weighted sampling for characters (must be after Drill Mode expansion)
            self.train_dataset.focus_chars = new_chars
            print(f"Epoch {epoch} | Koch Phase {phase}: Chars={current_chars} (20 Wpm, Clean) | Focus: {new_chars} | Focus Prob: {self.train_dataset.focus_prob}")

        elif phase == max_koch_phase + 1:
            # All characters (including prosigns), still clean
            self.train_dataset.min_wpm = 20
            self.train_dataset.max_wpm = 20
            self.train_dataset.min_snr = 50.0
            self.train_dataset.max_snr = 60.0
            self.train_dataset.jitter_max = 0.0
            self.train_dataset.weight_var = 0.0
            self.train_dataset.chars = self.train_dataset.all_chars
            self.train_dataset.min_len = 5
            self.train_dataset.max_len = 8 # OOM回避のため少し短縮
            self.train_dataset.focus_chars = None # No focus in full phase
            print(f"Epoch {epoch} | Phase {phase}: Full Chars, Clean")

        elif phase == max_koch_phase + 2:
            # Slight variations (No Fading yet)
            self.train_dataset.min_wpm = 18
            self.train_dataset.max_wpm = 25
            self.train_dataset.min_snr = 30.0
            self.train_dataset.max_snr = 45.0
            self.train_dataset.jitter_max = 0.03
            self.train_dataset.weight_var = 0.05
            self.train_dataset.fading_speed_min = 0.0
            self.train_dataset.fading_speed_max = 0.0
            self.train_dataset.chars = self.train_dataset.all_chars
            print(f"Epoch {epoch} | Phase {phase}: Full Chars, Slight Variations (No Fading)")

        elif phase == max_koch_phase + 3:
            # Practical SNR (15-25dB) with moderate fading
            self.train_dataset.min_snr = 15.0
            self.train_dataset.max_snr = 25.0
            self.train_dataset.min_fading = 0.2 # Start with fading to learn its pattern
            self.train_dataset.fading_speed_max = 0.1
            print(f"Epoch {epoch} | Phase {phase}: Practical SNR (15-25dB), Deep Fading (min_fading=0.2)")

        elif phase == max_koch_phase + 4:
            # Transition SNR (10-20dB)
            self.train_dataset.min_snr = 10.0
            self.train_dataset.max_snr = 20.0
            self.train_dataset.min_fading = 0.3 # Reduce fading depth as SNR drops
            print(f"Epoch {epoch} | Phase {phase}: Transition SNR (10-20dB), Moderate Fading (min_fading=0.3)")

        elif phase == max_koch_phase + 5:
            # Boundary SNR (5-15dB)
            self.train_dataset.min_snr = 5.0
            self.train_dataset.max_snr = 15.0
            self.train_dataset.min_fading = 0.4
            print(f"Epoch {epoch} | Phase {phase}: Boundary SNR (5-15dB), Weak Fading (min_fading=0.4)")

        elif phase == max_koch_phase + 6:
            # Zero SNR Entry (0-10dB)
            self.train_dataset.min_snr = 0.0
            self.train_dataset.max_snr = 10.0
            self.train_dataset.min_fading = 0.5
            print(f"Epoch {epoch} | Phase {phase}: Zero SNR Entry (0-10dB), Stable (min_fading=0.5)")

        elif phase == max_koch_phase + 7:
            # Negative SNR Entry (-5 to 5dB)
            self.train_dataset.min_snr = -5.0
            self.train_dataset.max_snr = 5.0
            self.train_dataset.min_fading = 0.6
            print(f"Epoch {epoch} | Phase {phase}: Negative SNR Entry (-5 to 5dB), Very Stable (min_fading=0.6)")

        elif phase == max_koch_phase + 8:
            # Target SNR: -8dB focus (-8 to 2dB)
            self.train_dataset.min_snr = -8.0
            self.train_dataset.max_snr = 2.0
            self.train_dataset.min_fading = 0.8
            self.train_dataset.fading_speed_max = 0.05
            print(f"Epoch {epoch} | Phase {phase}: Target SNR (-8 to 2dB), Rock Solid (min_fading=0.8)")

        elif phase == max_koch_phase + 9:
            # Deep Negative (-11 to -1dB)
            self.train_dataset.min_snr = -11.0
            self.train_dataset.max_snr = -1.0
            self.train_dataset.min_fading = 0.9
            print(f"Epoch {epoch} | Phase {phase}: Deep Negative SNR (-11 to -1dB), Rock Solid (min_fading=0.9)")

        elif phase == max_koch_phase + 10:
            # Deep Negative (-14 to -4dB)
            self.train_dataset.min_snr = -14.0
            self.train_dataset.max_snr = -4.0
            self.train_dataset.min_fading = 0.9
            print(f"Epoch {epoch} | Phase {phase}: Deep Negative SNR (-14 to -4dB), Rock Solid (min_fading=0.9)")

        elif phase == max_koch_phase + 11:
            # Human Limit Entry (-18 to -8dB)
            self.train_dataset.min_snr = -18.0
            self.train_dataset.max_snr = -8.0
            self.train_dataset.min_fading = 0.9
            print(f"Epoch {epoch} | Phase {phase}: Human Limit Entry (-18 to -8dB), Rock Solid (min_fading=0.9)")

        elif phase == max_koch_phase + 12:
            # Re-introduce fading at moderate SNR (-10 to 0dB)
            self.train_dataset.min_snr = -10.0
            self.train_dataset.max_snr = 0.0
            self.train_dataset.min_fading = 0.4
            self.train_dataset.fading_speed_max = 0.1
            print(f"Epoch {epoch} | Phase {phase}: Re-introducing Fading (-10 to 0dB), (min_fading=0.4)")

        elif phase == max_koch_phase + 13:
            # Deep Fading at Human Limit (-18 to -8dB)
            self.train_dataset.min_snr = -18.0
            self.train_dataset.max_snr = -8.0
            self.train_dataset.min_fading = 0.2
            self.train_dataset.fading_speed_max = 0.2
            print(f"Epoch {epoch} | Phase {phase}: Deep Fading at Human Limit (-18 to -8dB), (min_fading=0.2)")

        else:
            # True Extreme / Beyond Human Limit
            self.train_dataset.min_snr = -22.0
            self.train_dataset.max_snr = -12.0
            self.train_dataset.min_fading = 0.1
            self.train_dataset.fading_speed_max = 0.4
            print(f"Epoch {epoch} | Phase {phase}: True Extreme SNR (-22 to -12dB), Max Fading (min_fading=0.1)")

        # Apply same curriculum to validation dataset
        self.val_dataset.min_wpm = self.train_dataset.min_wpm
        self.val_dataset.max_wpm = self.train_dataset.max_wpm
        self.val_dataset.min_snr = self.train_dataset.min_snr
        self.val_dataset.max_snr = self.train_dataset.max_snr
        self.val_dataset.jitter_max = self.train_dataset.jitter_max
        self.val_dataset.weight_var = self.train_dataset.weight_var
        self.val_dataset.fading_speed_min = self.train_dataset.fading_speed_min
        self.val_dataset.fading_speed_max = self.train_dataset.fading_speed_max
        self.val_dataset.min_fading = self.train_dataset.min_fading
        self.val_dataset.chars = self.train_dataset.chars
        self.val_dataset.min_len = self.train_dataset.min_len
        self.val_dataset.max_len = self.train_dataset.max_len
        self.val_dataset.focus_chars = self.train_dataset.focus_chars
        self.val_dataset.focus_prob = self.train_dataset.focus_prob

        total_loss_accum = 0
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)
        for batch_idx, (waveforms, targets, lengths, target_lengths, _, _, signal_targets, boundary_targets) in enumerate(dataloader):
            # アライメント矯正のための「再加熱」ロジック
            # 停滞している場合（Epoch 60以降など）、一時的に LR を上げて局所解から脱出させる
            total_warmup_batches = len(dataloader)
            current_batch_global = (epoch - 1) * len(dataloader) + batch_idx
            
            if epoch >= 60 and epoch <= 70:
                # 再加熱期間: 初期 LR の 0.5倍程度を維持
                lr = self.args.lr * 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            elif current_batch_global < total_warmup_batches:
                lr = self.args.lr * (current_batch_global + 1) / total_warmup_batches
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            self.optimizer.zero_grad()
            mels, input_lengths = self.compute_mels_and_lengths(waveforms, lengths)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Forward
            # logits: (B, T, C)
            (logits, signal_logits, boundary_logits), _ = self.model(mels)
            
            # Ensure input_lengths does not exceed logits size
            input_lengths = torch.clamp(input_lengths, max=logits.size(1))
            
            loss, loss_dict = self.compute_loss(logits, signal_logits, boundary_logits, targets, target_lengths, input_lengths, signal_targets, boundary_targets)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Loss is {loss}, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            
            total_loss_accum += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} (CTC: {loss_dict['ctc'].item():.4f}, Sig: {loss_dict['sig'].item():.4f}, Bnd: {loss_dict['bound'].item():.4f}, Pnlty: {loss_dict['penalty'].item():.4f}) | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Debug lengths and logits
                with torch.no_grad():
                    # logits: (B, T, C)
                    blank_logits = logits[:, :, 0]
                    char_logits = logits[:, :, 1:]
                    max_char_logits, _ = char_logits.max(dim=-1)
                    
                    print(f"  Input Len (avg): {input_lengths.float().mean():.1f} | Target Len (avg): {target_lengths.float().mean():.1f}")
                    print(f"  Logits Stats | Blank Mean: {blank_logits.mean():.4f}, Max: {blank_logits.max():.4f}")
                    print(f"  Logits Stats | Chars Mean: {char_logits.mean():.4f}, Max: {max_char_logits.max():.4f}")

        return total_loss_accum / len(dataloader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_edit_distance = 0
        total_ref_length = 0
        
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        last_ref = ""
        last_hyp = ""
        timed_hyp = ""
        
        with torch.no_grad():
            for waveforms, targets, lengths, target_lengths, texts, wpms, signal_targets, boundary_targets in dataloader:
                mels, input_lengths = self.compute_mels_and_lengths(waveforms, lengths)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                (logits, signal_logits, boundary_logits), _ = self.model(mels)
                loss, _ = self.compute_loss(logits, signal_logits, boundary_logits, targets, target_lengths, input_lengths, signal_targets, boundary_targets)
                total_loss += loss.item()
                
                # デコード時はモデル自身の境界予測 (boundary_logits) でゲート制御を行う
                # ※調査中: 現在、モデルがバウンダリに過度に依存して判別をサボる傾向があるため、
                # ゲートを一時的に無効化して CTC 本来の性能を確認する。
                # bound_p = torch.sigmoid(boundary_logits)
                # gate_threshold = 0.05 if epoch < 10 else 0.2
                # is_boundary_pred = (bound_p > gate_threshold).squeeze(-1)
                
                preds = logits.argmax(dim=2)
                # preds[~is_boundary_pred] = 0
                
                sig_preds = signal_logits.argmax(dim=2)
                
                for i in range(preds.size(0)):
                    length = input_lengths[i].item()
                    p_indices = preds[i, :length]
                    p_sigs = sig_preds[i, :length]
                    
                    decoded_indices = []
                    decoded_positions = []
                    prev = -1
                    for t in range(len(p_indices)):
                        idx = p_indices[t].item()
                        if idx != 0 and idx != prev:
                            decoded_indices.append(idx)
                            decoded_positions.append(t)
                        prev = idx
                    
                    h_list = []
                    last_p = 0
                    for idx, pos in zip(decoded_indices, decoded_positions):
                        # 前の文字との間に単語間空白(5)があるかチェック
                        if any(p_sigs[last_p:pos] == 5):
                            h_list.append(" ")
                        h_list.append(ID_TO_CHAR.get(idx, ""))
                        last_p = pos
                        
                    hypothesis = "".join(h_list).strip()
                    reference = texts[i].strip()
                    
                    # CER計算からスペースを除外
                    reference_no_space = reference.replace(" ", "")
                    hypothesis_no_space = hypothesis.replace(" ", "")
                    
                    dist, _ = levenshtein(reference_no_space, hypothesis_no_space)
                    total_edit_distance += dist
                    total_ref_length += len(reference_no_space)
                    
                    last_ref = reference
                    last_hyp = hypothesis
                    timed_hyp = " ".join([f"{ID_TO_CHAR.get(idx, '')}({pos})" for idx, pos in zip(decoded_indices, decoded_positions)])
                    
        avg_loss = total_loss / len(dataloader)
        avg_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0.0
        print(f"Validation Epoch {epoch} | Avg Loss: {avg_loss:.4f} | CER: {avg_cer:.4f}")
        
        # 0:bg(_), 1:dit(#), 2:dah(=), 3:intra(.), 4:inter( ), 5:word( )
        class_map = {0: '_', 1: '#', 2: '=', 3: '.', 4: ' ', 5: ' '}
        sig_t_str = "".join([class_map.get(int(x), '?') for x in signal_targets[-1, :]])
        bound_t = boundary_targets[-1, :].cpu().numpy()
        bound_t_str = "".join(['!' if x > 0.5 else ' ' for x in bound_t])
        
        sig_h_preds = signal_logits[-1].argmax(dim=-1).cpu().numpy()
        sig_h_str = "".join([class_map.get(int(x), '?') for x in sig_h_preds])
        bound_h_probs = torch.sigmoid(boundary_logits[-1]).cpu().numpy().squeeze(-1)
        bound_h_str = "".join(['!' if x > 0.2 else ' ' for x in bound_h_probs])
        
        def overlay(sig, bound):
            return "".join([b if b == '!' else s for s, b in zip(sig, bound)])
            
        print(f"  Ref Sig:    {overlay(sig_t_str, bound_t_str)}")
        print(f"  Hyp Sig:    {overlay(sig_h_str, bound_h_str)}")
        print(f"  Bound Prob: Max={bound_h_probs.max():.4f}, Mean={bound_h_probs.mean():.4f}")
        print(f"  Sample Ref: {last_ref}")
        print(f"  Sample Hyp: {last_hyp}")
        print(f"  Timed Hyp:  {timed_hyp}")
        
        # アライメント崩壊の診断: 最後の文字が末尾付近に張り付いていないか
        if len(decoded_positions) > 0:
            last_pos = decoded_positions[-1]
            total_len = input_lengths[-1].item()
            if last_pos > total_len * 0.9:
                print(f"  WARNING: Alignment collapse detected! Last char at {last_pos}/{total_len}")

        return avg_loss, avg_cer

    def save_checkpoint(self, epoch, train_loss, val_loss, val_cer):
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        path = os.path.join(self.args.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'cer': val_cer,
            'curriculum_phase': self.current_phase,
            'phases_since_last_advance': self.phases_since_last_advance,
            'args': self.args
        }, path)
        print(f"Saved checkpoint: {path}")

        file_exists = os.path.isfile(self.history_path)
        with open(self.history_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_cer', 'lr', 'phase', 'stagnation'])
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_cer:.6f}",
                f"{self.optimizer.param_groups[0]['lr']:.8f}",
                self.current_phase,
                self.phases_since_last_advance
            ])

def main():
    parser = argparse.ArgumentParser(description="Train Streaming Conformer for CW")
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4) # Decreased LR for stability
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--d-model", type=int, default=config.D_MODEL)
    parser.add_argument("--n-head", type=int, default=config.N_HEAD)
    parser.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--curriculum-phase", type=int, default=0, help="Force specific curriculum phase (1, 2, 3). 0 for auto.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder parameters")
    parser.add_argument("--reset-ctc-head", action="store_true", help="Reset CTC head weights and bias")
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    start_epoch = 1
    resume_path = args.resume
    
    if resume_path == "latest":
        if os.path.isdir(args.save_dir):
            checkpoints = [f for f in os.listdir(args.save_dir) if f.endswith('.pt')]
            if checkpoints:
                # Sort by modification time or epoch number if possible
                # Assuming format checkpoint_epoch_X.pt
                def get_epoch(filename):
                    try:
                        return int(filename.split('_')[-1].split('.')[0])
                    except:
                        return 0
                checkpoints.sort(key=get_epoch)
                resume_path = os.path.join(args.save_dir, checkpoints[-1])
            else:
                print(f"No checkpoints found in '{args.save_dir}'")
                resume_path = None
        else:
            print(f"Directory '{args.save_dir}' does not exist")
            resume_path = None

    if resume_path:
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location=trainer.device)
            start_epoch = checkpoint['epoch'] + 1
            
            # Load curriculum state
            trainer.load_checkpoint_state(checkpoint)
            
            print(f"Resuming from Epoch {start_epoch}, Phase {trainer.current_phase}, Epochs in Phase: {trainer.phases_since_last_advance}")
            
            # Load model state dict with handling for size mismatch (e.g. vocab expansion)
            model_dict = trainer.model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            
            # Filter out unnecessary keys or keys with size mismatch
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            
            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            
            # Load the new state dict
            trainer.model.load_state_dict(model_dict)
            
            if args.reset_ctc_head:
                print("!!! RESETTING CTC HEAD !!!")
                nn.init.trunc_normal_(trainer.model.ctc_head.weight, std=0.01)
                nn.init.constant_(trainer.model.ctc_head.bias, 0)
                with torch.no_grad():
                    # 強力な Blank Penalty を初期バイアスとして設定
                    trainer.model.ctc_head.bias[0] = -10.0
                
                # Head をリセットした場合は optimizer もリセットして新しく学習させる
                trainer.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=args.lr, weight_decay=1e-5)
                print(f"Optimizer reset for fresh CTC head training. LR: {args.lr}")

            # Only load optimizer if we loaded the full model successfully (no vocab change)
            # If vocab changed, we should probably reset optimizer or at least the head params
            if len(pretrained_dict) == len(checkpoint['model_state_dict']):
                try:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Override learning rate with the one from command line
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = args.lr
                    print(f"Loaded full checkpoint '{resume_path}' (epoch {checkpoint['epoch']}) and set LR to {args.lr}")
                except ValueError as e:
                    print(f"Loaded model from '{resume_path}' (epoch {checkpoint['epoch']}), but failed to load optimizer state: {e}")
                    print("Optimizer state reset.")
            else:
                print(f"Loaded partial checkpoint '{resume_path}' (epoch {checkpoint['epoch']}). "
                      f"Optimizer state reset due to architecture change.")
                # We don't load optimizer state if model architecture changed
        else:
            print(f"No checkpoint found at '{resume_path}'")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss = trainer.train_epoch(epoch)
        val_loss, val_cer = trainer.validate(epoch)
        trainer.scheduler.step(val_cer)
        duration = time.time() - start_time
        print(f"Epoch {epoch} finished in {duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")
        trainer.update_curriculum(val_cer)
        trainer.save_checkpoint(epoch, train_loss, val_loss, val_cer)
        try:
            visualize_logs.plot_history(trainer.history_path, os.path.join(args.save_dir, "training_curves.png"))
        except:
            pass

if __name__ == "__main__":
    main()