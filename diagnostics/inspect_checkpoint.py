import os
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn

import config
from model import StreamingConformer

def format_size(num_params: int) -> str:
    """Format number of parameters for display."""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)

def inspect_checkpoint(checkpoint_path: str) -> None:
    """Load and display information about a checkpoint."""
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    # Load with weights_only=False to support older checkpoints if necessary, 
    # though AGENTS.md suggests we are in a controlled environment.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("\n=== Checkpoint Metadata ===")
    metadata_keys = ['epoch', 'loss', 'cer', 'curriculum_phase', 'phases_since_last_advance']
    for key in metadata_keys:
        if key in checkpoint:
            val = checkpoint[key]
            if isinstance(val, float):
                print(f"{key:25}: {val:.6f}")
            else:
                print(f"{key:25}: {val}")
        else:
            print(f"{key:25}: Not found")

    if 'args' in checkpoint:
        print("\n=== Training Arguments ===")
        args_dict = vars(checkpoint['args']) if hasattr(checkpoint['args'], '__dict__') else checkpoint['args']
        for k, v in args_dict.items():
            print(f"{k:25}: {v}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n=== Model Architecture (from state_dict) ===")
        
        # Infer architecture from state_dict if args not fully available or for verification
        # layers.0.ff1.0.weight -> (d_model * 4, d_model)
        d_model = 0
        num_layers = 0
        for key in state_dict.keys():
            if key.startswith("layers."):
                parts = key.split(".")
                layer_idx = int(parts[1])
                num_layers = max(num_layers, layer_idx + 1)
                if d_model == 0 and "ff1.0.weight" in key:
                    d_model = state_dict[key].shape[1]

        print(f"Inferred d_model          : {d_model}")
        print(f"Inferred num_layers       : {num_layers}")

        # Total parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters          : {format_size(total_params)}")

        # Layer-wise breakdown (summary)
        print("\n=== Parameter Breakdown ===")
        component_params: Dict[str, int] = {}
        for name, tensor in state_dict.items():
            base_name = name.split('.')[0]
            component_params[base_name] = component_params.get(base_name, 0) + tensor.numel()
        
        for name, count in sorted(component_params.items()):
            print(f"{name:25}: {format_size(count):>8}")

    else:
        print("\nError: 'model_state_dict' not found in checkpoint.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CW Decoder checkpoint.")
    parser.add_argument("checkpoint", type=str, nargs="?", default="latest", 
                        help="Path to checkpoint file or 'latest' (default: latest)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", 
                        help="Directory to search for 'latest' checkpoint (default: checkpoints)")
    
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    if checkpoint_path == "latest":
        if os.path.isdir(args.save_dir):
            checkpoints = [f for f in os.listdir(args.save_dir) if f.endswith('.pt')]
            if checkpoints:
                def get_epoch(filename):
                    try:
                        # Handle checkpoint_epoch_X.pt or checkpoint_epoch_X_final.pt
                        parts = filename.replace('_final', '').split('_')
                        return int(parts[-1].split('.')[0])
                    except:
                        return 0
                checkpoints.sort(key=get_epoch)
                checkpoint_path = os.path.join(args.save_dir, checkpoints[-1])
            else:
                print(f"No checkpoints found in '{args.save_dir}'")
                return
        else:
            print(f"Directory '{args.save_dir}' does not exist")
            return

    inspect_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()