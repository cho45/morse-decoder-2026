import torch
import torch.nn as nn
import onnxruntime as ort
import onnx
import numpy as np
import argparse
import config
from model import StreamingConformer
import os

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = config.NUM_LAYERS

    def forward(self, x: torch.Tensor, pcen_state: torch.Tensor, sub_cache: torch.Tensor,
                attn_k_0: torch.Tensor, attn_v_0: torch.Tensor, offset_0: torch.Tensor, conv_cache_0: torch.Tensor,
                attn_k_1: torch.Tensor, attn_v_1: torch.Tensor, offset_1: torch.Tensor, conv_cache_1: torch.Tensor,
                attn_k_2: torch.Tensor, attn_v_2: torch.Tensor, offset_2: torch.Tensor, conv_cache_2: torch.Tensor,
                attn_k_3: torch.Tensor, attn_v_3: torch.Tensor, offset_3: torch.Tensor, conv_cache_3: torch.Tensor):
        """
        Flattened forward for torch.export compatibility.
        """
        # States are always provided as tensors to avoid specialization.
        layer_states = [
            ((attn_k_0, attn_v_0, offset_0), conv_cache_0),
            ((attn_k_1, attn_v_1, offset_1), conv_cache_1),
            ((attn_k_2, attn_v_2, offset_2), conv_cache_2),
            ((attn_k_3, attn_v_3, offset_3), conv_cache_3),
        ]
        
        states = (pcen_state, sub_cache, layer_states)
        
        (logits, signal_logits, boundary_logits), (new_pcen_state, new_sub_cache, new_layer_states) = self.model(x, states)
        
        # Flatten new states
        res = [logits, signal_logits, boundary_logits, new_pcen_state, new_sub_cache]
        for (new_attn, new_conv) in new_layer_states:
            res.append(new_attn[0]) # k
            res.append(new_attn[1]) # v
            # offset is a Tensor or SymInt
            res.append(new_attn[2])
            res.append(new_conv)
            
        return tuple(res)

def export(checkpoint_path=None, output_path="cw_decoder.onnx"):
    device = torch.device("cpu")
    model = StreamingConformer(
        n_mels=config.N_BINS,
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS
    ).to(device)

    # Checkpoint があれば読み込む（なければランダム初期値でエクスポート）
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        # PyTorch 2.6+ compatibility: weights_only=False to allow loading arbitrary globals like argparse.Namespace
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Exporting with random weights.")
    
    model.eval()
    wrapper = ONNXWrapper(model)
    wrapper.eval()  # Ensure wrapper is in eval mode for correct ONNX export

    # Dummy inputs
    batch_size = 2
    seq_len = 12 # Multiple of 4, within dynamic_shapes [3, 20]
    # Use positive values for spectrogram dummy input to avoid NaNs in PCEN (log/pow)
    x = torch.rand(batch_size, seq_len, config.N_BINS)
    
    pcen_state = torch.zeros(batch_size, 1, config.N_BINS)
    sub_cache = torch.zeros(batch_size, 1, 2, config.N_BINS) # Initial sub_cache
    
    layer_states_flat = []
    # Use non-zero, non-one cache length for example inputs to avoid 0/1 specialization issues.
    example_cache_len = 10
    for _ in range(config.NUM_LAYERS):
        # attn_k, attn_v: (B, n_head, T_attn, d_k)
        d_k = config.D_MODEL // config.N_HEAD
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, example_cache_len, d_k)) # k
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, example_cache_len, d_k)) # v
        layer_states_flat.append(torch.tensor(example_cache_len, dtype=torch.long)) # offset (scalar tensor)
        # conv_cache: (B, d_model, kernel_size - 1)
        layer_states_flat.append(torch.zeros(batch_size, config.D_MODEL, config.KERNEL_SIZE - 1))

    input_names = ['x', 'pcen_state', 'sub_cache']
    output_names = ['logits', 'signal_logits', 'boundary_logits', 'new_pcen_state', 'new_sub_cache']
    
    for i in range(config.NUM_LAYERS):
        input_names.extend([f'attn_k_{i}', f'attn_v_{i}', f'offset_{i}', f'conv_cache_{i}'])
        output_names.extend([f'new_attn_k_{i}', f'new_attn_v_{i}', f'new_offset_{i}', f'new_conv_cache_{i}'])

    print(f"Exporting to {output_path}...")

    # [IMPORTANT] Use dynamo=True to use the new torch.export-based exporter.
    # This requires flattened inputs/outputs for the wrapper.
    
    # Define dynamic shapes for torch.export
    batch = torch.export.Dim("batch", min=1, max=4)
    # Use min=2 to avoid 0/1 specialization issues where possible
    seq = torch.export.Dim("seq", min=2, max=100)
    sub_cache_len = torch.export.Dim("sub_cache_len", min=0, max=100)
    attn_cache_len = torch.export.Dim("attn_cache_len", min=0, max=config.MAX_CACHE_LEN)
    
    # Define dynamic shapes for offsets as well.
    # For scalar tensors, an empty dict {} marks them as dynamic in torch.export.
    
    dynamic_shapes = {
        "x": {0: batch, 1: seq},
        "pcen_state": {0: batch},
        "sub_cache": {0: batch, 2: sub_cache_len},
    }
    for i in range(config.NUM_LAYERS):
        dynamic_shapes[f"attn_k_{i}"] = {0: batch, 2: attn_cache_len}
        dynamic_shapes[f"attn_v_{i}"] = {0: batch, 2: attn_cache_len}
        # For scalar tensors, an empty dict {} marks them as dynamic in torch.export.
        dynamic_shapes[f"offset_{i}"] = {}
        dynamic_shapes[f"conv_cache_{i}"] = {0: batch}

    # Step 1: Create an ExportedProgram with explicit constraints
    print("Creating ExportedProgram...")
    # Use strict=False to avoid over-specialization on example inputs (e.g. batch=1, cache=0)
    exported_program = torch.export.export(
        wrapper,
        args=(x, pcen_state, sub_cache, *layer_states_flat),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

    # Step 2: Convert ExportedProgram to ONNX
    print(f"Exporting ExportedProgram to {output_path}...")
    onnx_program = torch.onnx.export(
        exported_program,
        args=(), # ExportedProgram already contains example inputs
        f=None,  # Return ONNXProgram instead of writing to file directly
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=True
    )
    
    # Save the ONNX model using the ONNXProgram.save API
    # This is the recommended way to serialize models in PyTorch 2.6+
    onnx_program.save(output_path)

    print("Export complete.")

    # Verification
    print("Verifying ONNX model...")
    # Use CPU provider explicitly for consistency
    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    
    def to_numpy(t):
        return t.detach().cpu().numpy() if t is not None else None

    # Prepare inputs for ORT
    ort_inputs = {
        'x': to_numpy(x),
        'pcen_state': to_numpy(pcen_state),
        'sub_cache': to_numpy(sub_cache)
    }
    for i in range(config.NUM_LAYERS):
        ort_inputs[f'attn_k_{i}'] = to_numpy(layer_states_flat[i*4])
        ort_inputs[f'attn_v_{i}'] = to_numpy(layer_states_flat[i*4+1])
        ort_inputs[f'offset_{i}'] = to_numpy(layer_states_flat[i*4+2])
        ort_inputs[f'conv_cache_{i}'] = to_numpy(layer_states_flat[i*4+3])

    # PyTorch inference
    with torch.no_grad():
        pt_outputs = wrapper(x, pcen_state, sub_cache, *layer_states_flat)
    
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
    parser = argparse.ArgumentParser(description="Export StreamingConformer model to ONNX format")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="demo/cw_decoder.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    export(checkpoint_path=args.checkpoint, output_path=args.output)
