"""
ブラウザデモ（demo-mic.html 等）からエクスポートされたスペクトログラムデータ（JSON形式）に対して、
PyTorchのチェックポイントを使用して推論を実行するスクリプト。
ブラウザ側でのDSP処理結果が、学習済みモデルで正しく認識されるかを検証するために使用します。

使い方:
    python3 diagnostics/inference_exported_spec.py path/to/exported.json --checkpoint checkpoints/checkpoint_epoch_XXX.pt

引数:
    json: エクスポートされたJSONファイルのパス
    --checkpoint: 使用するPyTorchチェックポイントのパス（省略時は最新のものを自動選択）
    --device: 実行デバイス (cpu または cuda)
"""
import json
import numpy as np
import torch
import sys
import os
import argparse

# プロジェクトルートをパスに追加してモジュールをロード可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model import StreamingConformer
from stream_decode import CTCDecoder

def run_inference(json_path, checkpoint_path, device="cpu"):
    device = torch.device(device)
    
    # 1. Load Data
    with open(json_path, 'r') as f:
        export_data = json.load(f)
    
    print(f"--- Exported Data Info ---")
    print(f"Sample Rate: {export_data.get('sample_rate')} Hz")
    print(f"FFT Size: {export_data.get('n_fft')}")
    print(f"Hop Size: {export_data.get('hop_ms')} ms")
    print(f"--------------------------\n")

    spec_data = np.array([d['frame'] for d in export_data['data']], dtype=np.float32)
    spec_tensor = torch.from_numpy(spec_data).to(device) # (T, 14)
    
    # 2. Load Model from Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = StreamingConformer(
        n_mels=config.N_BINS,
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        n_head=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Initialize States and Decoder
    states = model.get_initial_states(1, device)
    decoder = CTCDecoder(config.ID_TO_CHAR)
    
    # 4. Inference Loop (Chunking like demo-mic.js)
    # demo-mic.js uses INFERENCE_CHUNK_SIZE = 10
    CHUNK_SIZE = 10
    T = spec_tensor.size(0)
    
    decoded_text = ""
    
    print(f"Processing {T} frames from {json_path} using {checkpoint_path}...")
    
    with torch.no_grad():
        for i in range(0, T, CHUNK_SIZE):
            chunk = spec_tensor[i:i+CHUNK_SIZE]
            if chunk.size(0) < CHUNK_SIZE:
                # Padding to match chunk size if needed
                pad = torch.zeros(CHUNK_SIZE - chunk.size(0), config.N_BINS, device=device)
                chunk = torch.cat([chunk, pad], dim=0)
            
            # Add batch dimension: (1, CHUNK_SIZE, 14)
            x = chunk.unsqueeze(0)
            
            (logits, sig_logits, boundary_logits), states = model(x, states)
            
            # Decode this chunk
            chunk_text = decoder.decode(logits, sig_logits, boundary_logits)
            if chunk_text:
                decoded_text += chunk_text
                sys.stdout.write(chunk_text)
                sys.stdout.flush()

    print(f"\n\n--- Final Decoded Result ---")
    print(decoded_text)
    print(f"----------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on exported spectrogram using PyTorch checkpoint")
    parser.add_argument("json", type=str, help="Path to exported JSON file")
    parser.add_argument("--checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Default checkpoint if not provided
    if not args.checkpoint:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("checkpoint_") and f.endswith(".pt")]
        if checkpoints:
            # Sort by modification time to get the truly latest one
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join("checkpoints", x)))
            args.checkpoint = os.path.join("checkpoints", checkpoints[-1])
        else:
            print("Error: No checkpoint found in checkpoints/ directory.")
            sys.exit(1)
            
    run_inference(args.json, args.checkpoint, args.device)