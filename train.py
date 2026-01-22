import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
from typing import List, Tuple

from data_gen import CWDataset, MORSE_DICT, generate_sample
from model import StreamingConformer
import config

# Use centralized config
CHARS = config.CHARS
CHAR_TO_ID = config.CHAR_TO_ID
ID_TO_CHAR = config.ID_TO_CHAR
NUM_CLASSES = config.NUM_CLASSES

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute Mel filterbank using config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            center=False # Crucial for streaming causality
        ).to(self.device)
        
        self.model = StreamingConformer(
            n_mels=config.N_MELS,
            num_classes=NUM_CLASSES,
            d_model=config.D_MODEL,
            n_head=config.N_HEAD,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Dataset
        self.train_dataset = CWDataset(num_samples=args.samples_per_epoch)
        self.val_dataset = CWDataset(num_samples=args.samples_per_epoch // 10)
        
    def collate_fn(self, batch: List[Tuple[torch.Tensor, str]]):
        # batch is a list of (waveform, text)
        waveforms = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        
        # Pad waveforms
        lengths = torch.tensor([w.shape[0] for w in waveforms])
        max_len = lengths.max()
        padded_waveforms = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            padded_waveforms[i, :w.shape[0]] = w
            
        # Extract Mel Spectrogram
        x = padded_waveforms.to(self.device)
        mels = self.mel_transform(x) # (B, F, T)
        mels = (mels + 1e-9).log()
        mels = mels.transpose(1, 2) # (B, T, F)
        
        # Subsampling in model is defined in config
        # Matches ConvSubsampling: (l_in - kernel_size) // stride + 1
        # where l_in = lengths // HOP_LENGTH, kernel_size = 3, stride = 2
        l_in = lengths // config.HOP_LENGTH
        # Initial padding in subsampling adds 2 frames if no cache
        l_in = l_in + 2
        input_lengths = (l_in - 3) // 2 + 1
        
        # Encode labels
        label_list = []
        label_lengths = []
        for text in texts:
            encoded = [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]
            label_list.extend(encoded)
            label_lengths.append(len(encoded))
            
        targets = torch.tensor(label_list, dtype=torch.long)
        target_lengths = torch.tensor(label_lengths, dtype=torch.long)
        
        return mels, targets, input_lengths, target_lengths, texts

    def train_epoch(self, epoch):
        self.model.train()
        
        # Curriculum: decrease min SNR as epochs progress
        # Start with 20-30 dB, end with 5-25 dB
        current_min_snr = max(5.0, 20.0 - (epoch - 1) * 3.0)
        current_max_snr = max(15.0, 30.0 - (epoch - 1) * 1.0)
        self.train_dataset.min_snr = current_min_snr
        self.train_dataset.max_snr = current_max_snr
        print(f"Curriculum SNR: {current_min_snr:.1f} - {current_max_snr:.1f} dB")

        total_loss = 0
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                shuffle=True, collate_fn=self.collate_fn)
        
        for batch_idx, (mels, targets, input_lengths, target_lengths, _) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward
            # logits: (B, T, C)
            logits, _ = self.model(mels)
            
            # Ensure input_lengths do not exceed logits.size(1)
            input_lengths = torch.clamp(input_lengths, max=logits.size(1))
            
            # CTC Loss expects (T, B, C)
            logits = logits.transpose(0, 1).log_softmax(2)
            
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Loss is {loss}, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        return total_loss / len(dataloader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, 
                                shuffle=False, collate_fn=self.collate_fn)
        
        with torch.no_grad():
            for mels, targets, input_lengths, target_lengths, texts in dataloader:
                logits, _ = self.model(mels)
                # Ensure input_lengths do not exceed logits.size(1)
                input_lengths = torch.clamp(input_lengths, max=logits.size(1))
                logits_t = logits.transpose(0, 1).log_softmax(2)
                loss = self.criterion(logits_t, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch, loss):
        path = os.path.join(self.args.save_dir, f"checkpoint_epoch_{epoch}.pt")
        os.makedirs(self.args.save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'args': self.args
        }, path)
        print(f"Saved checkpoint: {path}")

def main():
    parser = argparse.ArgumentParser(description="Train Streaming Conformer for CW")
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--d-model", type=int, default=config.D_MODEL)
    parser.add_argument("--n-head", type=int, default=config.N_HEAD)
    parser.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate(epoch)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch} finished in {duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if epoch % 1 == 0:
            trainer.save_checkpoint(epoch, val_loss)

if __name__ == "__main__":
    main()