import torch
import torchaudio
import numpy as np
import scipy.io.wavfile
import argparse
import time
import sys
import os
from typing import List, Tuple, Optional

from model import StreamingConformer
from data_gen import MORSE_DICT
import config
from inference_utils import decode_multi_task, preprocess_waveform

# Use centralized config
CHARS = config.CHARS
ID_TO_CHAR = config.ID_TO_CHAR
NUM_CLASSES = config.NUM_CLASSES

class CTCDecoder:
    def __init__(self, id_to_char):
        self.id_to_char = id_to_char
        self.last_id = 0 # Start with blank or 0
        
    def decode(self, logits: torch.Tensor, sig_logits: torch.Tensor, boundary_logits: torch.Tensor) -> str:
        """
        Greedy decoding of CTC logits.
        Spaces are now part of the CTC vocabulary.
        """
        # (1, T, C) -> (T, C)
        l = logits[0]
        
        # Extract predictions for this chunk
        preds = l.argmax(dim=-1)
        
        decoded_str = ""
        for t in range(len(preds)):
            idx = preds[t].item()
            
            if idx != self.last_id:
                if idx != 0:
                    char = self.id_to_char.get(idx, "")
                    decoded_str += char
                self.last_id = idx
        return decoded_str

class StreamDecoder:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load checkpoint to get args
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        train_args = checkpoint['args']
        
        self.model = StreamingConformer(
            n_mels=config.N_BINS,
            num_classes=NUM_CLASSES,
            d_model=config.D_MODEL,
            n_head=config.N_HEAD,
            num_layers=config.NUM_LAYERS,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.n_fft = config.N_FFT
        self.hop_length = config.HOP_LENGTH
        self.n_bins = config.N_BINS
        self.sample_rate = config.SAMPLE_RATE
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            center=False
        ).to(self.device)

        # Calculate bin indices for cropping
        self.bin_start = int(round(config.F_MIN * config.N_FFT / config.SAMPLE_RATE))
        self.bin_end = self.bin_start + config.N_BINS
        
        self.decoder = CTCDecoder(ID_TO_CHAR)
        self.states = (
            torch.zeros(1, 1, config.N_BINS).to(self.device), # pcen_state
            None, # sub_cache
            None  # layer_states
        )
        
        # Buffer for audio samples (including overlap for STFT)
        self.audio_buffer = np.array([], dtype=np.float32)

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Convert waveform to spectrogram.
        waveform: (T_raw,)
        """
        x = torch.from_numpy(waveform).to(self.device)
        # Note: preprocess_waveform handles padding, but for streaming chunks,
        # we don't want the full lookahead padding on every small chunk.
        # However, to ensure consistency in bin cropping and spectral shape,
        # we use the unified preprocess_waveform without the extra padding here
        # since streaming handles overlap via audio_buffer.
        
        # Standardized Preprocessing (Log scaling is now inside model's PCEN,
        # but for PyTorch inference we need to match the expected input)
        with torch.no_grad():
            if x.dim() == 1: x = x.unsqueeze(0)
            spec = self.spec_transform(x)
            spec = spec[:, self.bin_start:self.bin_end, :]
            # Current model expects raw spectrogram as PCEN is internal
            return spec.transpose(1, 2)

    def process_chunk(self, audio_chunk: np.ndarray, debug: bool = False):
        """
        Process a chunk of audio and print decoded characters.
        """
        # Combine with previous overlap
        combined = np.append(self.audio_buffer, audio_chunk)
        
        # Subsampling rate from config (2x)
        # We need at least N_FFT samples to get the first frame,
        # and then SUBSAMPLING_RATE * HOP_LENGTH samples for each subsequent model step.
        subsampling_samples = self.hop_length * config.SUBSAMPLING_RATE
        
        if len(combined) < self.n_fft:
            self.audio_buffer = combined
            return

        # Calculate how many model steps we can take
        # Total frames available: (len(combined) - n_fft) // hop + 1
        total_frames = (len(combined) - self.n_fft) // self.hop_length + 1
        num_steps = total_frames // config.SUBSAMPLING_RATE
        
        if num_steps == 0:
            self.audio_buffer = combined
            return

        # Number of frames to actually process
        n_frames = num_steps * config.SUBSAMPLING_RATE
        # Samples needed to produce exactly n_frames
        samples_needed = (n_frames - 1) * self.hop_length + self.n_fft
        
        chunk_to_process = combined[:samples_needed]
        # Keep the rest for next time. The next frame 0 will start after n_frames * hop_length samples
        self.audio_buffer = combined[num_steps * subsampling_samples:]
        
        with torch.no_grad():
            mels = self.preprocess(chunk_to_process)
            (logits, sig_logits, boundary_logits), self.states = self.model(mels, self.states)
                
            if debug:
                probs = torch.softmax(logits, dim=-1)
                max_probs, ids = torch.max(probs, dim=-1)
                ids = ids[0]
                max_probs = max_probs[0]
                if ids.sum() > 0:
                    # Print non-blank IDs for debugging
                    sys.stderr.write(f"\n[Debug] IDs: {ids.tolist()}, MaxProbs: {max_probs.tolist()}\n")
                    sys.stderr.flush()

            decoded = self.decoder.decode(logits, sig_logits, boundary_logits)
            if decoded:
                sys.stdout.write(decoded)
                sys.stdout.flush()

def run_file_mode(wav_path: str, decoder: StreamDecoder, debug: bool = False):
    sample_rate, waveform = scipy.io.wavfile.read(wav_path)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    
    print(f"Processing {wav_path} ({len(waveform)/sample_rate:.2f}s)...")
    
    # Simulate real-time by feeding chunks
    # Use default chunk duration (matches SR * chunk_duration = samples)
    # This is roughly 16 frames at 10ms hop.
    chunk_duration = (config.HOP_LENGTH * config.SUBSAMPLING_RATE * 8) / config.SAMPLE_RATE
    chunk_size = int(config.SAMPLE_RATE * chunk_duration)
    for i in range(0, len(waveform), chunk_size):
        chunk = waveform[i:i+chunk_size]
        decoder.process_chunk(chunk, debug=debug)
        if not debug:
            time.sleep(0.16) # Real-time simulation
    
    print("\nProcessing finished.")

def main():
    parser = argparse.ArgumentParser(description="Stream Decode Morse Code")
    parser.add_argument("--wav", type=str, help="Path to .wav file for file mode")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_1.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    decoder = StreamDecoder(args.checkpoint, device=args.device)
    
    if args.wav:
        run_file_mode(args.wav, decoder, debug=args.debug)
    else:
        print("No input source specified. Use --wav <path>.")
        # Microphone mode placeholder
        # try:
        #     import pyaudio
        #     ...
        # except ImportError:
        #     print("PyAudio not installed. Microphone mode unavailable.")

if __name__ == "__main__":
    main()