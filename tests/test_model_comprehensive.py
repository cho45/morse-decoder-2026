import pytest
import torch
import torch.nn as nn
import math
import config
from model import RelPositionalEncoding, CausalMultiHeadAttention, ConformerConvModule, ConformerBlock, ConvSubsampling, StreamingConformer

# --- Utilities ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tests for RelPositionalEncoding ---
def test_rel_pe_shapes():
    d_model = 64
    pe = RelPositionalEncoding(d_model)
    res = pe(length=100, offset=torch.tensor(0))
    assert res.shape == (100, d_model)
    
    res_offset = pe(length=50, offset=torch.tensor(100))
    assert res_offset.shape == (50, d_model)

def test_rel_pe_device():
    device = get_device()
    d_model = 64
    pe = RelPositionalEncoding(d_model).to(device)
    res = pe(length=100, offset=torch.tensor(0, device=device))
    assert res.device.type == device.type

# --- Tests for CausalMultiHeadAttention ---
def test_attention_causality():
    d_model = 64
    n_head = 4
    attn = CausalMultiHeadAttention(d_model, n_head)
    attn.eval()
    
    # Use positive inputs for PCEN-related components
    x = torch.rand(1, 100, d_model)
    with torch.no_grad():
        cache1 = (torch.zeros(1, n_head, 0, d_model // n_head), torch.zeros(1, n_head, 0, d_model // n_head), torch.tensor(0))
        out1, _ = attn(x, cache1)
        
        # Modify future frames (index 50+)
        x_modified = x.clone()
        x_modified[:, 50:, :] += 1.0
        cache2 = (torch.zeros(1, n_head, 0, d_model // n_head), torch.zeros(1, n_head, 0, d_model // n_head), torch.tensor(0))
        out2, _ = attn(x_modified, cache2)
        
    # Output before frame 50 must be identical
    assert torch.allclose(out1[:, :50, :], out2[:, :50, :], atol=1e-6)

def test_attention_streaming_consistency():
    d_model = 64
    n_head = 4
    attn = CausalMultiHeadAttention(d_model, n_head)
    attn.eval()
    
    # Use positive inputs for PCEN-related components
    x = torch.rand(1, 100, d_model)
    with torch.no_grad():
        cache_batch = (torch.zeros(1, n_head, 0, d_model // n_head), torch.zeros(1, n_head, 0, d_model // n_head), torch.tensor(0))
        y_batch, _ = attn(x, cache_batch)
        
        # Stream by 10-frame chunks
        states = (torch.zeros(1, n_head, 0, d_model // n_head), torch.zeros(1, n_head, 0, d_model // n_head), torch.tensor(0))
        y_streams = []
        for i in range(0, 100, 10):
            chunk = x[:, i:i+10, :]
            y_chunk, states = attn(chunk, states)
            y_streams.append(y_chunk)
        y_stream = torch.cat(y_streams, dim=1)
        
    assert torch.allclose(y_batch, y_stream, atol=1e-5)

# --- Tests for ConformerConvModule ---
def test_conv_module_streaming_consistency():
    d_model = 64
    conv = ConformerConvModule(d_model, kernel_size=31)
    conv.eval()
    
    # Use positive inputs for PCEN-related components
    x = torch.rand(1, 100, d_model)
    with torch.no_grad():
        cache_batch = torch.zeros(1, d_model, 30)
        y_batch, _ = conv(x, cache_batch)
        
        states = torch.zeros(1, d_model, 30)
        y_streams = []
        for i in range(0, 100, 10):
            chunk = x[:, i:i+10, :]
            y_chunk, states = conv(chunk, states)
            y_streams.append(y_chunk)
        y_stream = torch.cat(y_streams, dim=1)
        
    assert torch.allclose(y_batch, y_stream, atol=1e-5)

# --- Tests for ConvSubsampling ---
def test_subsampling_causality():
    in_channels = config.N_BINS
    out_channels = 64
    sub = ConvSubsampling(in_channels, out_channels)
    sub.eval()
    
    # Use positive inputs for PCEN-related components
    x = torch.rand(1, 100, in_channels)
    with torch.no_grad():
        cache1 = torch.zeros(1, 1, 2, in_channels)
        out1, _ = sub(x, cache1)
        
        # Modify future input
        x_mod = x.clone()
        x_mod[:, 80:, :] += 1.0
        cache2 = torch.zeros(1, 1, 2, in_channels)
        out2, _ = sub(x_mod, cache2)
        
    # Output is 2x downsampled. Frame 40 in output roughly corresponds to frame 80 in input.
    assert torch.allclose(out1[:, :35, :], out2[:, :35, :], atol=1e-6)

def test_subsampling_streaming_consistency():
    in_channels = config.N_BINS
    out_channels = 64
    sub = ConvSubsampling(in_channels, out_channels)
    sub.eval()
    
    # Needs to be multiple of 2 for this test to be simple
    # Use positive inputs for PCEN-related components
    x = torch.rand(1, 100, in_channels)
    with torch.no_grad():
        cache_batch = torch.zeros(1, 1, 2, in_channels)
        y_batch, _ = sub(x, cache_batch)
        
        states = torch.zeros(1, 1, 2, in_channels)
        y_streams = []
        # Non-even chunk sizes to test robustness
        chunks = [13, 7, 20, 40, 20]
        start = 0
        for sz in chunks:
            chunk = x[:, start:start+sz, :]
            y_chunk, states = sub(chunk, states)
            y_streams.append(y_chunk)
            start += sz
        y_stream = torch.cat(y_streams, dim=1)
        
    assert y_batch.shape == y_stream.shape
    assert torch.allclose(y_batch, y_stream, atol=1e-5)

# --- Integration Test for StreamingConformer ---
def test_model_full_streaming_consistency():
    device = get_device()
    model = StreamingConformer(
        n_mels=config.N_BINS,
        num_classes=10,
        d_model=64,
        num_layers=2
    ).to(device)
    model.eval()
    
    # Use positive inputs for PCEN
    x = torch.rand(1, 200, config.N_BINS).to(device)
    with torch.no_grad():
        states_batch = model.get_initial_states(1, device)
        (y_batch, _, _), _ = model(x, states_batch)
        
        # Start with None to match batch inference (initializes from first frame)
        states = model.get_initial_states(1, device)
        y_streams = []
        for i in range(0, 200, 20):
            chunk = x[:, i:i+20, :]
            (y_chunk, _, _), states = model(chunk, states)
            y_streams.append(y_chunk)
        y_stream = torch.cat(y_streams, dim=1)
        
    assert torch.allclose(y_batch, y_stream, atol=1e-3)

def test_model_parameter_count():
    # Ensure model is "Lightweight" as per plan.md
    model = StreamingConformer(d_model=144, num_layers=6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    # Should be roughly 2-5M for a lightweight Conformer
    assert total_params < 10e6 

if __name__ == "__main__":
    pytest.main([__file__])