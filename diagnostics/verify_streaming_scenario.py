import torch
import numpy as np
import sys
import os
import glob
from typing import List, Tuple

# プロジェクトルートをインポートパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen import TimingGenerator, WaveformGenerator, HFChannelSimulator
from stream_decode import StreamDecoder
import config

def verify_scenario():
    checkpoint_path = "checkpoints/checkpoint_epoch_2831.pt"
    if not os.path.exists(checkpoint_path):
        # 最新のチェックポイントを自動探索
        pts = glob.glob("checkpoints/*.pt")
        if not pts:
            print("No checkpoint found.")
            return
        checkpoint_path = sorted(pts, key=os.path.getmtime)[-1]
        print(f"Using latest checkpoint: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = StreamDecoder(checkpoint_path, device=device)

    sample_rate = config.SAMPLE_RATE
    timing_gen = TimingGenerator(sample_rate=sample_rate)
    waveform_gen = WaveformGenerator(sample_rate=sample_rate)
    simulator = HFChannelSimulator(sample_rate=sample_rate)

    text = "CQ CQ DE JH1UMV K <GAP:3.0> CQ CQ DE JH1UMV K <GAP:1.0>"
    
    wpms = [10, 15, 20, 30, 40][::-1]
    snrs = [-14, -12, -10, -5, 10, 40][::-1]

    print(f"Scenario: '{text}'")
    print(f"Sample Rate: {sample_rate} Hz")

    for snr in snrs:
        print(f"\n{'='*50}")
        print(f"Testing SNR: {snr} dB")
        print(f"{'='*50}")

        for wpm in wpms:
            # タイミング生成
            timing = timing_gen.generate_timing(text, wpm=wpm, max_duration=None)
            # クリーン波形生成 (frequency=700Hz)
            clean_waveform, _ = waveform_gen.generate_waveform(timing, frequency=700.0)

            # ノイズ付加
            noisy_waveform = simulator.apply_noise(clean_waveform, snr_2500=snr).astype(np.float32)
            
            # デコーダのステートをリセット
            decoder.states = decoder.model.get_initial_states(1, device=device)
            decoder.audio_buffer = np.array([], dtype=np.float32)
            decoder.decoder.last_id = 0
            
            # ストリーム推論のシミュレーション
            # 256ms 程度のチャンクで流し込む
            chunk_size = int(sample_rate * 0.256)
            
            decoded_text = ""
            
            # stdout への出力を一時的に横取りするために decoder.process_chunk をラップした関数を呼ぶか、
            # あるいは直接ロジックを回す。ここでは結果を蓄積したいのでロジックを少し模倣。
            
            print(f"{wpm}wpm: ", end="", flush=True)

            for i in range(0, len(noisy_waveform), chunk_size):
                chunk = noisy_waveform[i:i+chunk_size]
                
                # 以下、stream_decode.py の process_chunk から抜粋・改変
                combined = np.append(decoder.audio_buffer, chunk)
                subsampling_samples = decoder.hop_length * config.SUBSAMPLING_RATE
                
                if len(combined) < decoder.n_fft:
                    decoder.audio_buffer = combined
                    continue

                total_frames = (len(combined) - decoder.n_fft) // decoder.hop_length + 1
                num_steps = total_frames // config.SUBSAMPLING_RATE
                
                if num_steps == 0:
                    decoder.audio_buffer = combined
                    continue

                n_frames = num_steps * config.SUBSAMPLING_RATE
                samples_needed = (n_frames - 1) * decoder.hop_length + decoder.n_fft
                chunk_to_process = combined[:samples_needed]
                decoder.audio_buffer = combined[num_steps * subsampling_samples:]
                
                with torch.no_grad():
                    mels = decoder.preprocess(chunk_to_process)
                    (logits, sig_logits, boundary_logits), decoder.states = decoder.model(mels, decoder.states)
                    decoded = decoder.decoder.decode(logits, sig_logits, boundary_logits)
                    if decoded:
                        decoded_text += decoded
                        sys.stdout.write(decoded)
                        sys.stdout.flush()
            
            print()

if __name__ == "__main__":
    verify_scenario()
