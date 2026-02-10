import pytest
import torch
import tempfile
import os
import sys
import numpy as np
import onnxruntime as ort

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import StreamingConformer
from export_onnx import ONNXWrapper

def _export_model_to_onnx(model, onnx_path, seq_len=12):
    """Helper function to export model to ONNX format (duplicated from test_onnx_export.py for fixture usage)."""
    wrapper = ONNXWrapper(model)
    wrapper.eval()  # Ensure wrapper is in eval mode
    batch_size = 2
    # Use positive inputs for PCEN
    x = torch.rand(batch_size, seq_len, config.N_BINS)
    
    # Create initial states (simplified version of create_initial_states for export)
    d_k = config.D_MODEL // config.N_HEAD
    pcen_state = torch.zeros(batch_size, 1, config.N_BINS)
    sub_cache = torch.zeros(batch_size, 1, 2, config.N_BINS)
    
    layer_states_flat = []
    # Using example_cache_len=10 as in the original test
    example_cache_len = 10
    for _ in range(len(model.layers)):
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, example_cache_len, d_k))  # k
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, example_cache_len, d_k))  # v
        layer_states_flat.append(torch.tensor(example_cache_len, dtype=torch.long))  # offset
        layer_states_flat.append(torch.zeros(batch_size, config.D_MODEL, config.KERNEL_SIZE - 1))  # conv

    input_names = ['x', 'pcen_state', 'sub_cache']
    output_names = ['logits', 'signal_logits', 'boundary_logits', 'new_pcen_state', 'new_sub_cache']

    for i in range(len(model.layers)):
        input_names.extend([f'attn_k_{i}', f'attn_v_{i}', f'offset_{i}', f'conv_cache_{i}'])
        output_names.extend([f'new_attn_k_{i}', f'new_attn_v_{i}', f'new_offset_{i}', f'new_conv_cache_{i}'])

    # Define dynamic shapes for torch.export
    batch = torch.export.Dim("batch", min=1, max=4)
    seq = torch.export.Dim("seq", min=3, max=40)
    sub_cache_len = torch.export.Dim("sub_cache_len", min=0, max=100)
    attn_cache_len = torch.export.Dim("attn_cache_len", min=0, max=config.MAX_CACHE_LEN)
    
    dynamic_shapes = {
        "x": {0: batch, 1: seq},
        "pcen_state": {0: batch},
        "sub_cache": {0: batch, 2: sub_cache_len},
    }
    for i in range(config.NUM_LAYERS):
        dynamic_shapes[f"attn_k_{i}"] = {0: batch, 2: attn_cache_len}
        dynamic_shapes[f"attn_v_{i}"] = {0: batch, 2: attn_cache_len}
        dynamic_shapes[f"offset_{i}"] = {}
        dynamic_shapes[f"conv_cache_{i}"] = {0: batch}

    exported_program = torch.export.export(
        wrapper,
        args=(x, pcen_state, sub_cache, *layer_states_flat),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

    torch.onnx.export(
        exported_program,
        args=(),
        f=onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=True
    )

    return wrapper, input_names, output_names

@pytest.fixture(scope="module")
def shared_onnx_model():
    """
    Fixture to create and export the ONNX model once per module.
    Returns: (model, wrapper, session, input_names, output_names)
    """
    model = StreamingConformer(num_layers=config.NUM_LAYERS)
    model.eval()

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    try:
        wrapper, input_names, output_names = _export_model_to_onnx(model, onnx_path)
        # Create session once
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        yield model, wrapper, session, input_names, output_names
        
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)
