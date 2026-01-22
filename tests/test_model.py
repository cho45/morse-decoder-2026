import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import ConvSubsampling, ConformerConvModule, CausalRelMultiHeadAttention, RelPositionalEncoding, StreamingConformer

def check_streaming_consistency(module, input_shape, chunk_size, device='cpu'):
    module.eval()
    module.to(device)
    
    # Batch inference
    x = torch.randn(*input_shape).to(device)
    
    # Handle specific inputs for Attention
    if isinstance(module, CausalRelMultiHeadAttention):
        pos_enc = RelPositionalEncoding(config.D_MODEL)
        pos_emb = pos_enc(x.size(1))
        with torch.no_grad():
            y_batch, _ = module(x, pos_emb)
    elif isinstance(module, StreamingConformer):
        with torch.no_grad():
            y_batch, _ = module(x)
    else:
        with torch.no_grad():
            y_batch, _ = module(x)
            
    # Streaming inference
    states = None
    y_stream_list = []
    
    seq_len = input_shape[1]
    
    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            
            if isinstance(module, CausalRelMultiHeadAttention):
                # Retrieve current offset from states if available
                offset = states[2] if states is not None else 0
                chunk_pos_emb = pos_enc(chunk.size(1), offset=offset)
                y_chunk, states = module(chunk, chunk_pos_emb, states)
            elif isinstance(module, StreamingConformer):
                y_chunk, states = module(chunk, states)
            else:
                y_chunk, states = module(chunk, states)
                
            y_stream_list.append(y_chunk)
            
    y_stream = torch.cat(y_stream_list, dim=1)
    
    # Compare
    # Streaming output might be slightly shorter due to valid padding logic
    min_len = min(y_batch.size(1), y_stream.size(1))
    y_batch = y_batch[:, :min_len, :]
    y_stream = y_stream[:, :min_len, :]
    
    diff = torch.abs(y_batch - y_stream).max().item()
    return diff

@pytest.mark.parametrize("chunk_size", [1, 7, 40, 80, 33])
def test_conv_subsampling_consistency(chunk_size):
    # Input: (B, T, F) -> (1, 400, 80)
    input_shape = (1, 400, config.N_MELS)
    subsampling = ConvSubsampling(config.N_MELS, config.D_MODEL)
    
    diff = check_streaming_consistency(subsampling, input_shape, chunk_size)
    assert diff < 1e-5, f"ConvSubsampling consistency failed. Diff: {diff}"

@pytest.mark.parametrize("chunk_size", [1, 13, 20, 40])
def test_conformer_conv_consistency(chunk_size):
    # Input: (B, T, D) -> (1, 200, 144)
    input_shape = (1, 200, config.D_MODEL)
    conv_mod = ConformerConvModule(config.D_MODEL, config.KERNEL_SIZE)
    
    diff = check_streaming_consistency(conv_mod, input_shape, chunk_size)
    assert diff < 1e-5, f"ConformerConvModule consistency failed. Diff: {diff}"

@pytest.mark.parametrize("chunk_size", [1, 13, 20, 40])
def test_attention_consistency(chunk_size):
    # Input: (B, T, D) -> (1, 200, 144)
    input_shape = (1, 200, config.D_MODEL)
    attn = CausalRelMultiHeadAttention(config.D_MODEL, config.N_HEAD)
    
    diff = check_streaming_consistency(attn, input_shape, chunk_size)
    assert diff < 1e-5, f"Attention consistency failed. Diff: {diff}"

@pytest.mark.parametrize("chunk_size", [40, 80, 33])
def test_model_consistency(chunk_size):
    input_shape = (1, 400, config.N_MELS)
    model = StreamingConformer(num_layers=2) # Small model for testing
    
    diff = check_streaming_consistency(model, input_shape, chunk_size)
    assert diff < 1e-5, f"StreamingConformer consistency failed. Diff: {diff}"

def test_cache_limit_and_pe_consistency():
    """Test if cache limit works and PE index remains valid."""
    d_model = 144
    n_head = 4
    max_cache = config.MAX_CACHE_LEN
    
    model = StreamingConformer(d_model=d_model, n_head=n_head, num_layers=1)
    model.eval()
    
    # Process enough chunks to exceed MAX_CACHE_LEN
    chunk_size = 100
    num_chunks = (max_cache // chunk_size) + 5
    
    states = None
    with torch.no_grad():
        for i in range(num_chunks):
            x = torch.randn(1, chunk_size, config.N_MELS)
            logits, states = model(x, states)
            
            # Check attn cache size of the first layer
            k_cache = states[1][0][0][0] # states -> (sub_cache, layer_states[0] -> (attn_cache -> (k, v, offset), conv_cache))
            assert k_cache.size(2) <= max_cache
            
            # Check offset is correctly managed
            offset = states[1][0][0][2]
            assert offset <= max_cache