import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import config
from model import StreamingConformer
import os

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = len(model.layers)

    def forward(self, x, sub_cache, *layer_states_flat):
        """
        x: (B, T, F)
        sub_cache: (B, 1, T_sub, F)
        layer_states_flat: [attn_k_0, attn_v_0, offset_0, conv_cache_0, attn_k_1, ...]
        """
        layer_states = []
        for i in range(0, len(layer_states_flat), 4):
            k = layer_states_flat[i]
            v = layer_states_flat[i+1]
            offset = layer_states_flat[i+2] # Keep as tensor
            conv = layer_states_flat[i+3]
            layer_states.append(((k, v, offset), conv))
        
        states = (sub_cache, layer_states)
        
        (logits, signal_logits, boundary_logits), (new_sub_cache, new_layer_states) = self.model(x, states)
        
        # Flatten new states
        new_states_flat = [new_sub_cache]
        for (new_attn, new_conv) in new_layer_states:
            new_states_flat.append(new_attn[0]) # k
            new_states_flat.append(new_attn[1]) # v
            new_states_flat.append(new_attn[2].view(())) # offset (0-dim tensor)
            new_states_flat.append(new_conv)
            
        return logits, signal_logits, boundary_logits, *new_states_flat

def export():
    device = torch.device("cpu")
    model = StreamingConformer(
        n_mels=config.N_MELS,
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS
    ).to(device)
    
    # Checkpoint があれば読み込む（なければランダム初期値でエクスポート）
    checkpoint_path = "checkpoint_epoch_145_clean_all.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("No checkpoint found, exporting with random weights.")
    
    model.eval()
    wrapper = ONNXWrapper(model)
    wrapper.eval()  # Ensure wrapper is in eval mode for correct ONNX export

    # Dummy inputs
    batch_size = 1
    seq_len = 40 # Multiple of 4
    x = torch.randn(batch_size, seq_len, config.N_MELS)
    
    sub_cache = torch.zeros(batch_size, 1, 2, config.N_MELS) # Initial sub_cache
    
    layer_states_flat = []
    for _ in range(config.NUM_LAYERS):
        # attn_k, attn_v: (B, n_head, T_attn, d_k)
        d_k = config.D_MODEL // config.N_HEAD
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, 0, d_k)) # k
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, 0, d_k)) # v
        layer_states_flat.append(torch.tensor(0, dtype=torch.long)) # offset (0-dim)
        # conv_cache: (B, d_model, kernel_size - 1)
        layer_states_flat.append(torch.zeros(batch_size, config.D_MODEL, config.KERNEL_SIZE - 1))

    input_names = ['x', 'sub_cache']
    output_names = ['logits', 'signal_logits', 'boundary_logits', 'new_sub_cache']
    
    dynamic_axes = {
        'x': {0: 'batch_size', 1: 'seq_len'},
        'sub_cache': {0: 'batch_size', 2: 'sub_cache_len'},
        'logits': {0: 'batch_size', 1: 'out_seq_len'},
        'signal_logits': {0: 'batch_size', 1: 'out_seq_len'},
        'boundary_logits': {0: 'batch_size', 1: 'out_seq_len'},
        'new_sub_cache': {0: 'batch_size', 2: 'new_sub_cache_len'},
    }

    for i in range(config.NUM_LAYERS):
        input_names.extend([f'attn_k_{i}', f'attn_v_{i}', f'offset_{i}', f'conv_cache_{i}'])
        output_names.extend([f'new_attn_k_{i}', f'new_attn_v_{i}', f'new_offset_{i}', f'new_conv_cache_{i}'])
        
        dynamic_axes[f'attn_k_{i}'] = {0: 'batch_size', 2: 'attn_cache_len'}
        dynamic_axes[f'attn_v_{i}'] = {0: 'batch_size', 2: 'attn_cache_len'}
        dynamic_axes[f'conv_cache_{i}'] = {0: 'batch_size'}
        
        dynamic_axes[f'new_attn_k_{i}'] = {0: 'batch_size', 2: 'new_attn_cache_len'}
        dynamic_axes[f'new_attn_v_{i}'] = {0: 'batch_size', 2: 'new_attn_cache_len'}
        dynamic_axes[f'new_conv_cache_{i}'] = {0: 'batch_size'}

    onnx_path = "cw_decoder.onnx"
    print(f"Exporting to {onnx_path}...")
    
    torch.onnx.export(
        wrapper,
        (x, sub_cache, *layer_states_flat),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17, # Using a recent opset
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL
    )
    print("Export complete.")

    # Verification
    print("Verifying ONNX model...")
    # Use CPU provider explicitly for consistency
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    def to_numpy(t):
        return t.detach().cpu().numpy() if t is not None else None

    # Prepare inputs for ORT
    ort_inputs = {
        'x': to_numpy(x),
        'sub_cache': to_numpy(sub_cache)
    }
    for i in range(config.NUM_LAYERS):
        ort_inputs[f'attn_k_{i}'] = to_numpy(layer_states_flat[i*4])
        ort_inputs[f'attn_v_{i}'] = to_numpy(layer_states_flat[i*4+1])
        ort_inputs[f'offset_{i}'] = to_numpy(layer_states_flat[i*4+2])
        ort_inputs[f'conv_cache_{i}'] = to_numpy(layer_states_flat[i*4+3])

    # PyTorch inference
    with torch.no_grad():
        pt_outputs = wrapper(x, sub_cache, *layer_states_flat)
    
    # ORT inference
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare
    for i, name in enumerate(output_names):
        pt_out = to_numpy(pt_outputs[i])
        ort_out = ort_outputs[i]
        diff = np.abs(pt_out - ort_out).max()
        print(f"Output '{name}' max diff: {diff:.6e}")
        if diff > 1e-4:
            print(f"WARNING: Large difference in '{name}'!")
        else:
            print(f"Verification passed for '{name}'.")

if __name__ == "__main__":
    export()