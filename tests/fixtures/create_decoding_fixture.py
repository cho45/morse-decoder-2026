import torch
import numpy as np
import json
import os
import sys
from model import StreamingConformer
from data_gen import generate_sample
from inference_utils import preprocess_waveform
import config

# Use the same logic as visualize_snr_performance.py
def create_fixture():
    checkpoint_path = "checkpoints/checkpoint_epoch_614_final.pt"
    if not os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        if not checkpoints:
            print("No checkpoints found.")
            return
        checkpoint_path = os.path.join("checkpoints", checkpoints[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = StreamingConformer(
        n_mels=config.N_BINS,
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        n_head=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cases = [
        {"name": "standard_25wpm", "text": "CQ CQ DE JH1UMV K", "wpm": 25, "snr": 40},
        {"name": "fast_40wpm", "text": "CQ CQ DE JH1UMV K", "wpm": 40, "snr": 40},
        {"name": "noisy_20wpm", "text": "CQ CQ DE JH1UMV K", "wpm": 20, "snr": 0},
    ]
    
    fixtures = {}

    for case in cases:
        print(f"Generating fixture for {case['name']}...")
        # EXACTLY as in visualize_snr_performance.py evaluate_batch
        freq = 700.0
        waveform, _, _, _ = generate_sample(
            text=case['text'], wpm=case['wpm'], snr_2500=case['snr'], frequency=freq,
            jitter=0.0, weight=1.0, fading_speed=0.0, min_fading=1.0,
            qrm_prob=0.1, impulse_prob=0.001
        )
        
        mels = preprocess_waveform(waveform, device)
        
        with torch.no_grad():
            (logits, sig_logits, bound_logits), _ = model(mels)
            
        # Save raw logits
        logits_list = logits[0].cpu().numpy().tolist()
        sig_logits_list = sig_logits[0].cpu().numpy().tolist()
        bound_logits_list = bound_logits[0].cpu().numpy().tolist()
        
        fixtures[case['name']] = {
            "text": case['text'],
            "wpm": case['wpm'],
            "snr": case['snr'],
            "logits": logits_list,
            "sig_logits": sig_logits_list,
            "bound_logits": bound_logits_list,
        }

    fixture_path = "tests/fixtures/decoding_test.json"
    with open(fixture_path, "w") as f:
        json.dump(fixtures, f, indent=2)
        
    print(f"Fixtures created at {fixture_path}")

if __name__ == "__main__":
    os.makedirs("tests/fixtures", exist_ok=True)
    create_fixture()