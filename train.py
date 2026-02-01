import torch
import torch.nn as nn
import torch.nn.functional as F
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
from curriculum import CurriculumManager
from inference_utils import decode_multi_task, calculate_cer, map_prosigns

# Use centralized config
CHARS = config.CHARS
CHAR_TO_ID = config.CHAR_TO_ID
ID_TO_CHAR = config.ID_TO_CHAR
NUM_CLASSES = config.NUM_CLASSES

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute Spectrogram transform using config
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0,
            center=False # Use False for strict streaming causality
        ).to(self.device)
        
        # Calculate bin indices for cropping
        # 16000 / 512 = 31.25 Hz/bin
        # 500 / 31.25 = 16.0
        # 900 / 31.25 = 28.8
        self.bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
        self.bin_end = self.bin_start + config.N_BINS
        print(f"Spectrogram Bin Range: {self.bin_start} to {self.bin_end} ({config.N_BINS} bins)")

        self.model = StreamingConformer(
            n_mels=config.N_BINS,
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
        # 0: Background/Space, 1: Dit, 2: Dah, 3: Inter-word space
        # クラス不均衡を考慮しつつ、偽陽性を抑えるため Inter-word space (3) の重みを調整
        # Dit(1)は短く見落としやすいため 1.5 に強化。
        # Word Space(3)は過剰分割を防ぐため 1.5 に緩和。
        sig_weights = torch.tensor([0.9, 1.9, 1.0, 1.5]).to(self.device)
        self.signal_criterion = nn.CrossEntropyLoss(weight=sig_weights, reduction='none')
        # Binary Cross Entropy for Character Boundary Detection
        # Positive weight to handle class imbalance (boundaries are rare)
        # 偽陽性（過剰分割）を抑え、安定した境界検出を行うため pos_weight を 5.0 に調整
        pos_weight = torch.tensor([5.0]).to(self.device)
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        
        # Dataset
        self.train_dataset = CWDataset(num_samples=args.samples_per_epoch)
        self.val_dataset = CWDataset(num_samples=args.samples_per_epoch // 10)
        
        self.history_path = os.path.join(args.save_dir, "history.csv")
        
        # Adaptive Curriculum State
        self.curriculum = CurriculumManager()
        self.current_phase = 1
        self.phases_since_last_advance = 0
        self.min_epochs_per_phase = 2
        self.cer_threshold_to_advance = 0.01

        # Initialize phase from args if provided
        if self.args.curriculum_phase > 0:
            self.current_phase = self.args.curriculum_phase

    def update_curriculum(self, val_cer):
        """
        Update curriculum phase based on validation CER.
        """
        # Always increment counter
        self.phases_since_last_advance += 1
        
        # Logic: If CER is low enough AND we've spent enough time in this phase
        if val_cer < self.cer_threshold_to_advance and self.phases_since_last_advance >= self.min_epochs_per_phase:
            max_phase = self.curriculum.get_max_phase()
            
            if self.current_phase < max_phase:
                print(f"*** PERFORMANCE GOOD (CER {val_cer:.4f} < {self.cer_threshold_to_advance}). ADVANCING TO PHASE {self.current_phase + 1} ***")
                self.current_phase += 1
                self.phases_since_last_advance = 0
                
                # Reset optimizer LR to boost learning for new phase
                reset_lr = self.args.lr
                print(f"[Curriculum] Resetting Learning Rate to {reset_lr}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = reset_lr
            else:
                print(f"*** MAX PHASE REACHED. CONTINUING REFINEMENT ***")
        else:
            print(f"[DEBUG] Phase {self.current_phase}: CER {val_cer:.4f} (Target < {self.cer_threshold_to_advance}), Epochs {self.phases_since_last_advance}/{self.min_epochs_per_phase} - WAITING")
            
            # 停滞打破のための LR 再加熱 (Re-heating)
            # 15エポック以上停滞したら、LR を初期値まで戻して局所解からの脱出を試みる
            # 以降、5エポックごとに再加熱を繰り返す。
            # 極限環境 (-15dB) への挑戦では、より慎重な再加熱を行う。
            if self.phases_since_last_advance >= 15 and (self.phases_since_last_advance - 15) % 5 == 0:
                # 25エポック以上停滞した場合は初期値の 1.5倍まで LR を引き上げる。
                multiplier = 1.5 if self.phases_since_last_advance >= 25 else 1.0
                reheat_lr = self.args.lr * multiplier
                print(f"*** STAGNATION DETECTED ({self.phases_since_last_advance} epochs). RE-HEATING LR TO {reheat_lr} ***")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = reheat_lr

    def load_checkpoint_state(self, checkpoint):
        """Load curriculum state from checkpoint."""
        if self.args.curriculum_phase == 0 and 'curriculum_phase' in checkpoint:
            self.current_phase = checkpoint['curriculum_phase']
        
        if 'phases_since_last_advance' in checkpoint:
            self.phases_since_last_advance = checkpoint['phases_since_last_advance']
            
    def collate_fn(self, batch: List[Tuple[torch.Tensor, str, int, torch.Tensor, torch.Tensor, bool]]):
        # batch is a list of (waveform, text, wpm, signal_labels, boundary_labels, is_phrase)
        waveforms, texts, wpms, signal_labels, boundary_labels, is_phrases = zip(*batch)
        
        # 物理的な先読み（Lookahead）アライメント:
        # Causal Model（過去しか見ない）で先読みを実現するために、
        # 入力波形の後ろにパディングを追加し、同時に正解ラベル（Signal/Boundary）を
        # 「遅延」させる（右にシフトする）。
        # これにより、時刻 T の入力に対して、時刻 T - delay の正解を予測することになり、
        # 実質的に T - delay の予測のために T までの未来情報を使うことになる。
        
        lookahead_samples = config.LOOKAHEAD_FRAMES * config.HOP_LENGTH
        
        # Pad waveforms (入力はそのまま、後ろにパディング)
        lengths = torch.tensor([w.shape[0] for w in waveforms])
        padded_waveforms = torch.zeros(len(waveforms), lengths.max() + lookahead_samples)
        for i, w in enumerate(waveforms):
            padded_waveforms[i, :w.shape[0]] = w
            
        # Pad & Shift labels
        # signal_labels は Melフレーム単位。
        # モデル出力は Subsampling されているため、シフト量も Subsampling 後のフレーム数で計算する。
        
        # 1. 全体の Mel フレーム数
        l_total = padded_waveforms.size(1)
        l_mel = (l_total - config.N_FFT) // config.HOP_LENGTH + 1
        
        # 2. Subsampling 後のフレーム数 (モデル出力サイズ)
        padding_t = 2 # From compute_mels_and_lengths
        input_lengths_val = (l_mel + padding_t - 3) // 2 + 1
        
        # 3. シフト量 (Lookahead frames converted to output domain)
        # config.LOOKAHEAD_FRAMES は Mel フレーム単位
        shift_frames = config.LOOKAHEAD_FRAMES // config.SUBSAMPLING_RATE
        
        padded_signals = torch.zeros(len(signal_labels), input_lengths_val, dtype=torch.long)
        padded_boundaries = torch.zeros(len(boundary_labels), input_lengths_val, dtype=torch.float32)
        
        for i, (s, b) in enumerate(zip(signal_labels, boundary_labels)):
            # Downsample labels
            s_downsampled = s[::config.SUBSAMPLING_RATE]
            b_downsampled = b[::config.SUBSAMPLING_RATE]
            
            # Shift labels to the right
            # [Pad(shift)] + [Label]
            # 有効なラベルの長さ
            valid_len = min(len(s_downsampled), input_lengths_val - shift_frames)
            
            if valid_len > 0:
                padded_signals[i, shift_frames : shift_frames + valid_len] = s_downsampled[:valid_len].long()
                padded_boundaries[i, shift_frames : shift_frames + valid_len] = b_downsampled[:valid_len].float()

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
                    # Always include characters that are in the vocabulary (including space)
                    if text[i] in CHAR_TO_ID:
                        tokens.append(text[i])
                    i += 1
            encoded = [CHAR_TO_ID[t] for t in tokens]
            label_list.extend(encoded)
            label_lengths.append(len(encoded))
            
        return padded_waveforms, torch.tensor(label_list, dtype=torch.long), lengths, torch.tensor(label_lengths, dtype=torch.long), texts, torch.tensor(wpms), padded_signals, padded_boundaries, torch.tensor(is_phrases)

    def compute_mels_and_lengths(self, waveforms, lengths):
        x = waveforms.to(self.device)
        spec = self.spec_transform(x) # (B, F, T)
        
        # Crop frequency bins
        spec = spec[:, self.bin_start:self.bin_end, :] # (B, N_BINS, T)

        # PCEN will handle scaling and normalization inside the model
        mels = spec.transpose(1, 2) # (B, T, F)
        
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

    def compute_loss(self, logits, signal_logits, boundary_logits, targets, target_lengths, input_lengths, signal_targets, boundary_targets, penalty_weight=2.0):
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
        
        # [設計意図] A/W などのプレフィックス識別を維持しつつ、ハイフン等の長符号の学習を促すため、
        # ペナルティ制約に「遊び（Smoothing）」を導入する。
        # [設計意図] A/W などのプレフィックス識別を維持しつつ、ハイフン等の長符号の学習を促すため、
        # ペナルティ制約に「遊び（Smoothing）」を導入する。
        # バウンダリの周辺数フレームを許容範囲として広げる。
        # 毎バッチ計算のオーバーヘッドを避けるため、シンプルな 1D Max Pool のみを使用。
        with torch.no_grad():
            # (B, T) -> (B, 1, T)
            soft_bound = bound_t.unsqueeze(1)
            # 左右 7フレーム程度を許容 (計 15フレーム)
            # フェーズ進展に伴う境界の曖昧さを許容し、Deletion を抑制するために拡大
            soft_bound = F.max_pool1d(soft_bound, kernel_size=15, stride=1, padding=7)
            soft_bound = soft_bound.squeeze(1)
            
        # ペナルティ対象： バウンダリ（およびその周辺）でない場所
        penalty_mask = (1.0 - soft_bound)
        
        # 数学的修正: 全フレーム数(mask.sum())で割ると、10秒のデータではペナルティが1/1000に希釈されてしまう。
        # サンプルあたりのペナルティとして機能させるため、バッチサイズ(logits.size(0))で正規化する。
        illegal_spike_loss = (char_probs_sum * penalty_mask * mask).sum() / (logits.size(0) + 1e-6)

        # 3. Signal Detection Loss
        sig_t = signal_targets[:, :signal_logits.size(1)].to(self.device)
        raw_sig_loss = self.signal_criterion(signal_logits.transpose(1, 2), sig_t)
        sig_loss = (raw_sig_loss * mask).sum() / (mask.sum() + 1e-6)
        
        # 4. Boundary Detection Loss
        bound_t_bce = boundary_targets[:, :boundary_logits.size(1)].to(self.device).unsqueeze(-1)
        raw_bound_loss = self.boundary_criterion(boundary_logits, bound_t_bce).squeeze(-1)
        bound_loss = (raw_bound_loss * mask).sum() / (mask.sum() + 1e-6)

        # 損失の重みバランスを調整:
        # 信号認識 (Sig Loss) の重みを 10.0 に落ち着かせ、CTC の重みを 2.0 に強化。
        # これにより、信号の物理構造を維持しつつ、文字識別能力の向上を図る。
        # 低 SNR 環境下でのアライメント崩壊を防ぐため、Illegal Spike Penalty は 0.5倍で維持。
        total_loss = 2.0 * ctc_loss + 10.0 * sig_loss + 1.0 * bound_loss + (penalty_weight * 0.5) * illegal_spike_loss
        
        return total_loss, {'ctc': ctc_loss, 'sig': sig_loss, 'bound': bound_loss, 'penalty': illegal_spike_loss}

    def train_epoch(self, epoch):
        self.model.train()
        
        # Fetch current curriculum phase
        p = self.curriculum.get_phase(self.current_phase)
        
        # Apply curriculum to train dataset
        self.train_dataset.chars = p.chars
        self.train_dataset.min_snr_2500 = p.min_snr_2500
        self.train_dataset.max_snr_2500 = p.max_snr_2500
        self.train_dataset.min_wpm = p.min_wpm
        self.train_dataset.max_wpm = p.max_wpm
        self.train_dataset.jitter_max = p.jitter
        self.train_dataset.weight_var = p.weight_var
        self.train_dataset.phrase_prob = p.phrase_prob
        self.train_dataset.focus_prob = p.focus_prob
        self.train_dataset.fading_speed_min = p.fading_speed[0]
        self.train_dataset.fading_speed_max = p.fading_speed[1]
        self.train_dataset.min_fading = p.min_fading
        self.train_dataset.drift_prob = p.drift_prob
        self.train_dataset.qrn_prob = p.qrn_prob
        self.train_dataset.qrm_prob = p.qrm_prob
        self.train_dataset.impulse_prob = p.impulse_prob
        self.train_dataset.agc_prob = p.agc_prob
        self.train_dataset.multipath_prob = p.multipath_prob
        self.train_dataset.clipping_prob = p.clipping_prob
        self.train_dataset.min_gain_db = p.min_gain_db
        
        # VRAM Safe length limits (10s fixed buffer handles most cases, but we keep text length reasonable)
        self.train_dataset.min_len = 5
        self.train_dataset.max_len = 15 if p.phrase_prob > 0 else 10
        
        # focus_chars は curriculum phase から取得する
        self.train_dataset.focus_chars = p.focus_chars

        print(f"Epoch {epoch} | Phase {self.current_phase} ({p.name})")
        print(f"  Chars: {p.chars} | Focus: {self.train_dataset.focus_chars}")
        print(f"  Env: SNR_2500={p.min_snr_2500:.1f}-{p.max_snr_2500:.1f}dB, WPM={p.min_wpm}-{p.max_wpm}, Jitter={p.jitter}, WeightVar={p.weight_var}, Gain={p.min_gain_db}dB")
        print(f"  Aug: Fading={p.min_fading} (spd {p.fading_speed}), Drift={p.drift_prob}, QRN={p.qrn_prob}, AGC={p.agc_prob}, Multipath={p.multipath_prob}, Clipping={p.clipping_prob}")
        print(f"  Prob: Phrase={p.phrase_prob}, Focus={p.focus_prob} | Penalty: {p.penalty_weight}")

        # Apply same curriculum to validation dataset
        self.val_dataset.min_wpm = self.train_dataset.min_wpm
        self.val_dataset.max_wpm = self.train_dataset.max_wpm
        self.val_dataset.min_snr_2500 = self.train_dataset.min_snr_2500
        self.val_dataset.max_snr_2500 = self.train_dataset.max_snr_2500
        self.val_dataset.jitter_max = self.train_dataset.jitter_max
        self.val_dataset.weight_var = self.train_dataset.weight_var
        self.val_dataset.fading_speed_min = self.train_dataset.fading_speed_min
        self.val_dataset.fading_speed_max = self.train_dataset.fading_speed_max
        self.val_dataset.min_fading = self.train_dataset.min_fading
        self.val_dataset.min_gain_db = self.train_dataset.min_gain_db
        self.val_dataset.drift_prob = getattr(self.train_dataset, 'drift_prob', 0.0)
        self.val_dataset.qrn_prob = getattr(self.train_dataset, 'qrn_prob', 0.0)
        self.val_dataset.qrm_prob = getattr(self.train_dataset, 'qrm_prob', 0.1)
        self.val_dataset.impulse_prob = getattr(self.train_dataset, 'impulse_prob', 0.001)
        self.val_dataset.agc_prob = getattr(self.train_dataset, 'agc_prob', 0.0)
        self.val_dataset.multipath_prob = getattr(self.train_dataset, 'multipath_prob', 0.0)
        self.val_dataset.clipping_prob = getattr(self.train_dataset, 'clipping_prob', 0.0)
        self.val_dataset.chars = self.train_dataset.chars
        self.val_dataset.min_len = self.train_dataset.min_len
        self.val_dataset.max_len = self.train_dataset.max_len
        self.val_dataset.focus_chars = self.train_dataset.focus_chars
        self.val_dataset.focus_prob = self.train_dataset.focus_prob
        self.val_dataset.phrase_prob = self.train_dataset.phrase_prob
        self.val_dataset.min_freq = self.train_dataset.min_freq
        self.val_dataset.max_freq = self.train_dataset.max_freq

        total_loss_accum = 0
        
        # Statistics for the epoch
        actual_wpms = []
        actual_lens = []

        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)
        for batch_idx, (waveforms, targets, lengths, target_lengths, _, wpms, signal_targets, boundary_targets, _) in enumerate(dataloader):
            actual_wpms.extend(wpms.tolist())
            actual_lens.extend(target_lengths.tolist())
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

            # 勾配累積の処理
            mels, input_lengths = self.compute_mels_and_lengths(waveforms, lengths)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Forward
            states = self.model.get_initial_states(mels.size(0), mels.device)
            (logits, signal_logits, boundary_logits), _ = self.model(mels, states)
            input_lengths = torch.clamp(input_lengths, max=logits.size(1))
            
            loss, loss_dict = self.compute_loss(logits, signal_logits, boundary_logits, targets, target_lengths, input_lengths, signal_targets, boundary_targets, penalty_weight=p.penalty_weight)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Loss is {loss}, skipping batch")
                continue
                
            # 累積ステップ数で正規化
            loss = loss / self.args.accumulation_steps
            loss.backward()
            
            # 指定ステップごとに更新
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss_accum += loss.item() * self.args.accumulation_steps
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item() * self.args.accumulation_steps:.4f} (CTC: {loss_dict['ctc'].item():.4f}, Sig: {loss_dict['sig'].item():.4f}, Bnd: {loss_dict['bound'].item():.4f}, Pnlty: {loss_dict['penalty'].item():.4f}) | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Debug lengths and logits
                with torch.no_grad():
                    # logits: (B, T, C)
                    blank_logits = logits[:, :, 0]
                    char_logits = logits[:, :, 1:]
                    max_char_logits, _ = char_logits.max(dim=-1)
                    
                    print(f"  Input Len (avg): {input_lengths.float().mean():.1f} | Target Len (avg): {target_lengths.float().mean():.1f}")
                    print(f"  Logits Stats | Blank Mean: {blank_logits.mean():.4f}, Max: {blank_logits.max():.4f}")
                    print(f"  Logits Stats | Chars Mean: {char_logits.mean():.4f}, Max: {max_char_logits.max():.4f}")

        # Print epoch statistics
        if actual_wpms and actual_lens:
            print(f"  Actual Stats | WPM: {min(actual_wpms):.1f}-{max(actual_wpms):.1f} | Len: {min(actual_lens)}-{max(actual_lens)}")

        return total_loss_accum / len(dataloader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_edit_distance = 0
        total_ref_length = 0
        
        # Levenshtein ops stats
        total_matches = 0
        total_subs = 0
        total_ins = 0
        total_dels = 0

        # シグナル分類精度計測用
        total_sig_correct = 0
        total_sig_elements = 0

        # 分離計測用
        total_dist_phrase = 0
        total_len_phrase = 0
        total_dist_random = 0
        total_len_random = 0
        
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        last_ref = ""
        last_hyp = ""
        timed_hyp = ""
        
        with torch.no_grad():
            for waveforms, targets, lengths, target_lengths, texts, wpms, signal_targets, boundary_targets, is_phrases in dataloader:
                mels, input_lengths = self.compute_mels_and_lengths(waveforms, lengths)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                states = self.model.get_initial_states(mels.size(0), mels.device)
                (logits, signal_logits, boundary_logits), _ = self.model(mels, states)
                loss, _ = self.compute_loss(logits, signal_logits, boundary_logits, targets, target_lengths, input_lengths, signal_targets, boundary_targets)
                total_loss += loss.item()
                
                sig_preds = signal_logits.argmax(dim=2)
                
                # シグナル分類精度の計算 (Mask を考慮)
                input_lengths_clamped = torch.clamp(input_lengths, max=signal_logits.size(1))
                for i in range(signal_logits.size(0)):
                    length = input_lengths_clamped[i].item()
                    sig_p = sig_preds[i, :length]
                    sig_t = signal_targets[i, :length].to(self.device)
                    total_sig_correct += (sig_p == sig_t).sum().item()
                    total_sig_elements += length

                for i in range(logits.size(0)):
                    length = input_lengths[i].item()
                    b_probs = torch.sigmoid(boundary_logits[i, :length]).squeeze(-1)
                    
                    # Unified Decoding
                    hypothesis, timed_output = decode_multi_task(
                        logits[i, :length],
                        signal_logits[i, :length],
                        b_probs
                    )
                    
                    reference = texts[i]
                    # Use map_prosigns to count prosigns as single characters, but keep spaces
                    ref_len = len(map_prosigns(reference))
                    
                    # Unified CER calculation
                    dist_rate = calculate_cer(reference, hypothesis)
                    dist = dist_rate * ref_len
                    total_edit_distance += dist
                    total_ref_length += ref_len

                    if is_phrases[i]:
                        total_dist_phrase += dist
                        total_len_phrase += ref_len
                    else:
                        total_dist_random += dist
                        total_len_random += ref_len

                    last_ref = reference
                    last_hyp = hypothesis
                    timed_hyp = " ".join([f"{char}({pos})" for char, pos in timed_output])
                    
        avg_loss = total_loss / len(dataloader)
        avg_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0.0
        cer_phrase = total_dist_phrase / total_len_phrase if total_len_phrase > 0 else 0.0
        cer_random = total_dist_random / total_len_random if total_len_random > 0 else 0.0
        
        avg_sig_acc = total_sig_correct / total_sig_elements if total_sig_elements > 0 else 0.0
        print(f"Validation Epoch {epoch} | Avg Loss: {avg_loss:.4f} | CER: {avg_cer:.4f} (Phrase: {cer_phrase:.4f}, Random: {cer_random:.4f}) | Sig Acc: {avg_sig_acc:.4f}")
        # 0:bg/space(_), 1:dit(#), 2:dah(=), 3:word( )
        class_map = {0: '_', 1: '#', 2: '=', 3: ' '}
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
        
        # アライメント崩壊の診断: 信号の終了位置に対して、出力が不自然に遅延（末尾へ溜め込み）していないか
        if len(timed_output) > 0:
            last_pos = timed_output[-1][1]
            total_len = input_lengths[-1].item()
            
            # 実際の信号の最終位置を特定 (1:Dit, 2:Dah)
            sig_t = signal_targets[-1, :total_len]
            sig_indices = torch.where((sig_t == 1) | (sig_t == 2))[0]
            if len(sig_indices) > 0:
                actual_sig_end = sig_indices[-1].item()
                # 信号終了から 1.5秒 (150フレーム) 以上遅れている場合は崩壊とみなす
                if last_pos > actual_sig_end + 150:
                    print(f"  WARNING: Alignment collapse detected! Sig end: {actual_sig_end}, Last char: {last_pos}, Total: {total_len}")
            elif last_pos > total_len * 0.9:
                # 信号がない（異常系）場合のフォールバック
                print(f"  WARNING: Alignment collapse detected (No signal)! Last char: {last_pos}")

        return avg_loss, avg_cer, cer_phrase, cer_random, avg_sig_acc

    def save_checkpoint(self, epoch, train_loss, val_loss, val_cer, cer_phrase, cer_random, sig_acc):
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
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_cer', 'cer_phrase', 'cer_random', 'sig_acc', 'lr', 'phase', 'stagnation'])
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_cer:.6f}",
                f"{cer_phrase:.6f}",
                f"{cer_random:.6f}",
                f"{sig_acc:.6f}",
                f"{self.optimizer.param_groups[0]['lr']:.8f}",
                self.current_phase,
                self.phases_since_last_advance
            ])

def main():
    parser = argparse.ArgumentParser(description="Train Streaming Conformer for CW")
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4) # Fine-tuning LR
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients before update")
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

    # 勾配累積がサンプル数と整合しているかチェック (数学的厳密性の確保)
    if args.samples_per_epoch % (args.batch_size * args.accumulation_steps) != 0:
        raise ValueError(f"samples_per_epoch ({args.samples_per_epoch}) must be divisible by "
                         f"batch_size * accumulation_steps ({args.batch_size * args.accumulation_steps})")
    
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
            checkpoint = torch.load(resume_path, map_location=trainer.device, weights_only=False)
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
        val_loss, val_cer, cer_phrase, cer_random, sig_acc = trainer.validate(epoch)
        trainer.scheduler.step(val_cer)
        duration = time.time() - start_time
        print(f"Epoch {epoch} finished in {duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} (P: {cer_phrase:.4f}, R: {cer_random:.4f}) | Sig Acc: {sig_acc:.4f}")
        trainer.update_curriculum(val_cer)
        trainer.save_checkpoint(epoch, train_loss, val_loss, val_cer, cer_phrase, cer_random, sig_acc)
        try:
            visualize_logs.plot_history(trainer.history_path, os.path.join(args.save_dir, "training_curves.png"))
        except:
            pass

if __name__ == "__main__":
    main()