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

# Use centralized config
CHARS = config.CHARS
CHAR_TO_ID = config.CHAR_TO_ID
ID_TO_CHAR = config.ID_TO_CHAR
NUM_CLASSES = config.NUM_CLASSES

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    
    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
            
    return current[n]

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
            d_model=config.D_MODEL,
            n_head=config.N_HEAD,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(self.device)

        if args.freeze_encoder:
            print("Freezing encoder parameters...")
            for name, param in self.model.named_parameters():
                if "ctc_head" not in name:
                    param.requires_grad = False
        
        # 余計なハイパーパラメータ引数への依存を減らし、標準的な AdamW 設定を使用
        # 学習率は warmup 後に args.lr に到達するように調整
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=1e-5)
        
        # 指標を CER に変更し、忍耐強めに設定
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
        # CTC Loss: zero_infinity=True to handle edge cases in early training
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        
        # Dataset
        self.train_dataset = CWDataset(num_samples=args.samples_per_epoch)
        self.val_dataset = CWDataset(num_samples=args.samples_per_epoch // 10)
        
        self.history_path = os.path.join(args.save_dir, "history.csv")
        
    def collate_fn(self, batch: List[Tuple[torch.Tensor, str, int]]):
        # batch is a list of (waveform, text, wpm)
        waveforms = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        wpms = [b[2] for b in batch]
        
        # Physical Lookahead Alignment:
        # Add LOOKAHEAD_FRAMES at the end of each waveform to allow the causal model
        # to "look ahead" into the future signal for the current label.
        lookahead_samples = config.LOOKAHEAD_FRAMES * config.HOP_LENGTH
        
        # Pad waveforms
        lengths = torch.tensor([w.shape[0] for w in waveforms])
        # Add lookahead only at the end.
        max_len = lengths.max() + lookahead_samples
        padded_waveforms = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            # Place signal at the beginning
            padded_waveforms[i, :w.shape[0]] = w
            
        # Return padded waveforms on CPU
        # Mel transform will be done in train_epoch/validate
        
        # Subsampling logic remains same, but we need to calculate input_lengths based on
        # expected Mel frames.
        # l_in = lengths // config.HOP_LENGTH
        
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
                        if token in CHAR_TO_ID:
                            tokens.append(token)
                            i = end + 1
                            continue
                
                # Check for multi-char tokens like CQ, DE
                found = False
                for token in config.PROSIGNS:
                    if text.startswith(token, i):
                        tokens.append(token)
                        i += len(token)
                        found = True
                        break
                if found: continue

                if text[i] in CHAR_TO_ID:
                    tokens.append(text[i])
                i += 1
            
            encoded = [CHAR_TO_ID[t] for t in tokens]
            label_list.extend(encoded)
            label_lengths.append(len(encoded))
            
        targets = torch.tensor(label_list, dtype=torch.long)
        target_lengths = torch.tensor(label_lengths, dtype=torch.long)
        
        return padded_waveforms, targets, lengths, target_lengths, texts, torch.tensor(wpms)

    def compute_mels_and_lengths(self, waveforms, lengths):
        x = waveforms.to(self.device)
        mels = self.mel_transform(x) # (B, F, T)
        
        # Log scaling and simple robust normalization
        # Shift and scale so that silence is around -1 and signal is around 1-3.
        mels = (torch.log1p(mels * 100.0) - 2.0) / 2.0
        
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

    def train_epoch(self, epoch):
        self.model.train()
        
        # Koch Method Curriculum
        # Standard Koch order: K, M, R, S, T, L, O, P, I, A, N, W, G, D, U, X, Z, Q, V, F, Y, C, H, J, B, L, 7, 5, 1, 2, 9, 0, 8, 3, 4, 6
        # (Simplified and adjusted to match our vocabulary)
        KOCH_CHARS = "KMRSTLOPANWGDUXZQVFHYCJB7512908346/?"
        
        if self.args.curriculum_phase > 0:
            phase = self.args.curriculum_phase
        else:
            # Auto-calculate phase based on epoch
            # Every 2 epochs, add 2 more characters from Koch order
            # Until all 38 chars are added (around epoch 38)
            phase = (epoch + 1) // 2
            if phase > 19: # All Koch chars added
                if epoch <= 50: phase = 20 # Expansion / Clean
                elif epoch <= 70: phase = 21 # Slight variations
                else: phase = 22 # Realistic

        if phase <= 19:
            # Koch Phases: Gradually increase character set
            num_chars = min(2 + (phase - 1) * 2, len(KOCH_CHARS))
            current_chars = KOCH_CHARS[:num_chars]
            
            self.train_dataset.min_wpm = 20
            self.train_dataset.max_wpm = 20
            self.train_dataset.min_snr = 100.0 # Clean
            self.train_dataset.max_snr = 100.0
            self.train_dataset.jitter_max = 0.0
            self.train_dataset.weight_var = 0.0
            self.train_dataset.chars = current_chars
            self.train_dataset.min_len = 2
            self.train_dataset.max_len = 6
            print(f"Epoch {epoch} | Koch Phase {phase}: Chars={current_chars} (20 WPM, Clean)")

        elif phase == 20:
            # All characters (including prosigns), still clean
            self.train_dataset.min_wpm = 20
            self.train_dataset.max_wpm = 20
            self.train_dataset.min_snr = 50.0
            self.train_dataset.max_snr = 60.0
            self.train_dataset.jitter_max = 0.0
            self.train_dataset.weight_var = 0.0
            self.train_dataset.chars = self.train_dataset.all_chars
            self.train_dataset.min_len = 5
            self.train_dataset.max_len = 10
            print(f"Epoch {epoch} | Phase 20: Full Chars, Clean")

        elif phase == 21:
            # Slight variations
            self.train_dataset.min_wpm = 18
            self.train_dataset.max_wpm = 25
            self.train_dataset.min_snr = 30.0
            self.train_dataset.max_snr = 45.0
            self.train_dataset.jitter_max = 0.03
            self.train_dataset.weight_var = 0.05
            self.train_dataset.chars = self.train_dataset.all_chars
            print(f"Epoch {epoch} | Phase 21: Full Chars, Slight Variations")

        else:
            # Realistic
            self.train_dataset.min_wpm = 10
            self.train_dataset.max_wpm = 45
            self.train_dataset.min_snr = 5.0
            self.train_dataset.max_snr = 30.0
            self.train_dataset.jitter_max = 0.10
            self.train_dataset.weight_var = 0.15
            self.train_dataset.chars = self.train_dataset.all_chars
            print(f"Epoch {epoch} | Phase 22: Realistic (Noise & Jitter)")

        # Apply same curriculum to validation dataset
        self.val_dataset.min_wpm = self.train_dataset.min_wpm
        self.val_dataset.max_wpm = self.train_dataset.max_wpm
        self.val_dataset.min_snr = self.train_dataset.min_snr
        self.val_dataset.max_snr = self.train_dataset.max_snr
        self.val_dataset.jitter_max = self.train_dataset.jitter_max
        self.val_dataset.weight_var = self.train_dataset.weight_var
        self.val_dataset.chars = self.train_dataset.chars
        self.val_dataset.min_len = self.train_dataset.min_len
        self.val_dataset.max_len = self.train_dataset.max_len

        total_loss = 0
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                shuffle=True, collate_fn=self.collate_fn)
        
        for batch_idx, (waveforms, targets, lengths, target_lengths, _, _) in enumerate(dataloader):
            # Simple LR Warmup for the first epoch
            # Warmup over 1 epoch (about 42 batches) to get moving quickly.
            total_warmup_batches = len(dataloader)
            current_batch_global = (epoch - 1) * len(dataloader) + batch_idx
            if current_batch_global < total_warmup_batches:
                lr = self.args.lr * (current_batch_global + 1) / total_warmup_batches
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            self.optimizer.zero_grad()
            
            mels, input_lengths = self.compute_mels_and_lengths(waveforms, lengths)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward
            # logits: (B, T, C)
            logits, _ = self.model(mels)
            
            # Ensure input_lengths does not exceed logits size
            input_lengths = torch.clamp(input_lengths, max=logits.size(1))
            
            # CTC Loss expects (T, B, C)
            logits_t = logits.transpose(0, 1).log_softmax(2)
            
            loss = self.criterion(logits_t, targets, input_lengths, target_lengths)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Loss is {loss}, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                # Debug lengths and logits
                with torch.no_grad():
                    # logits: (B, T, C)
                    blank_logits = logits[:, :, 0]
                    char_logits = logits[:, :, 1:]
                    max_char_logits, _ = char_logits.max(dim=-1)
                    
                    print(f"  Input Len (avg): {input_lengths.float().mean():.1f} | Target Len (avg): {target_lengths.float().mean():.1f}")
                    print(f"  Logits Stats | Blank Mean: {blank_logits.mean():.4f}, Max: {blank_logits.max():.4f}")
                    print(f"  Logits Stats | Chars Mean: {char_logits.mean():.4f}, Max: {max_char_logits.max():.4f}")
                    
                    # Check if model is stuck in blank-only output
                    if blank_logits.mean() > max_char_logits.mean() + 2.0:
                        print("  WARNING: Model is heavily biased towards BLANK.")
                
                
        return total_loss / len(dataloader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_edit_distance = 0
        total_ref_length = 0
        
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size,
                                shuffle=False, collate_fn=self.collate_fn)
        
        last_ref = ""
        last_hyp = ""

        with torch.no_grad():
            for waveforms, targets, lengths, target_lengths, texts, wpms in dataloader:
                mels, input_lengths = self.compute_mels_and_lengths(waveforms, lengths)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                logits, _ = self.model(mels)
                
                input_lengths = torch.clamp(input_lengths, max=logits.size(1))

                logits_t = logits.transpose(0, 1).log_softmax(2)
                loss = self.criterion(logits_t, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Calculate CER
                preds = logits.argmax(dim=2) # (B, T)
                for i in range(preds.size(0)):
                    # Only decode up to input_length to avoid padded region
                    length = input_lengths[i].item() if input_lengths.dim() > 0 else input_lengths.item()
                    pred_indices = preds[i, :length]
                    
                    # Standard greedy decode for CTC
                    decoded_indices = []
                    decoded_positions = []
                    prev = -1
                    for t in range(pred_indices.size(0)):
                        idx = pred_indices[t].item()
                        if idx != 0 and idx != prev:
                            decoded_indices.append(idx)
                            decoded_positions.append(t)
                        prev = idx
                    
                    # Reconstruct hypothesis with spaces based on temporal gaps between characters.
                    # Use actual WPM of the sample for precise thresholding.
                    wpm = wpms[i].item()
                    dot_len_sec = 1.2 / wpm
                    frame_duration = (config.HOP_LENGTH * config.SUBSAMPLING_RATE) / config.SAMPLE_RATE
                    
                    # Threshold for word space (usually 7 dots, char space is 3 dots)
                    # We use 5.0 dots as the threshold to robustly separate char space from word space.
                    threshold_frames = (5.0 * dot_len_sec) / frame_duration

                    hypothesis = ""
                    if len(decoded_indices) > 0:
                        for j in range(len(decoded_indices)):
                            char = ID_TO_CHAR.get(decoded_indices[j], "")
                            if j > 0:
                                # In CTC, decoded_positions[j] is the peak of the character.
                                # The gap between peaks is a good proxy for the space between characters.
                                gap = decoded_positions[j] - decoded_positions[j-1]
                                if gap > threshold_frames:
                                    hypothesis += " "
                            hypothesis += char
                    
                    hypothesis = hypothesis.strip()
                    timed_hyp = " ".join([f"{ID_TO_CHAR.get(idx, '')}({pos})" for idx, pos in zip(decoded_indices, decoded_positions)])
                    # For CER calculation, we focus on the character sequence.
                    # Since spaces are not in the vocabulary, we compare sequences without spaces.
                    reference = texts[i].replace(" ", "").strip()
                    hyp_no_space = hypothesis.replace(" ", "").strip()
                    
                    dist = levenshtein(reference, hyp_no_space)
                    total_edit_distance += dist
                    total_ref_length += len(reference)
                    
                    last_ref = reference
                    last_hyp = hypothesis

        avg_loss = total_loss / len(dataloader)
        avg_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0.0
        print(f"Validation Epoch {epoch} | Avg Loss: {avg_loss:.4f} | CER: {avg_cer:.4f}")
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
        
        # Save Model Checkpoint
        path = os.path.join(self.args.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'cer': val_cer,
            'args': self.args
        }, path)
        print(f"Saved checkpoint: {path}")

        # Save History to CSV
        file_exists = os.path.isfile(self.history_path)
        with open(self.history_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_cer', 'lr'])
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_cer:.6f}",
                f"{self.optimizer.param_groups[0]['lr']:.8f}"
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
        
        if epoch % 1 == 0:
            trainer.save_checkpoint(epoch, train_loss, val_loss, val_cer)

if __name__ == "__main__":
    main()