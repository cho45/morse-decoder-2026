import torch
import sys
import os

def inspect(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"--- Checkpoint: {path} ---")
        print(f"Epoch: {checkpoint.get('epoch')}")
        print(f"Phase: {checkpoint.get('curriculum_phase')}")
        print(f"Phases Since Advance: {checkpoint.get('phases_since_last_advance')}")
        print(f"Loss: {checkpoint.get('loss')}")
        print(f"CER: {checkpoint.get('cer')}")
        
        # Check optimizer LR
        opt_state = checkpoint.get('optimizer_state_dict', {})
        if 'param_groups' in opt_state:
            print(f"LR: {opt_state['param_groups'][0]['lr']}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        # Find latest
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if checkpoints:
            def get_epoch(filename):
                try:
                    return int(filename.split('_')[-1].split('.')[0])
                except:
                    return 0
            checkpoints.sort(key=get_epoch)
            latest = checkpoints[-1]
            inspect(os.path.join('checkpoints', latest))
