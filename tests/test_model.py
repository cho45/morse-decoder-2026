import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import ConvSubsampling, ConformerConvModule, CausalMultiHeadAttention, RelPositionalEncoding, StreamingConformer

def check_streaming_consistency(module, input_shape, chunk_size, device='cpu'):
    module.eval()
    module.to(device)
    
    batch_size = input_shape[0]
    
    # Create initial states based on module type
    if isinstance(module, CausalMultiHeadAttention):
        initial_states = (
            torch.zeros(batch_size, module.n_head, 0, module.d_k, device=device),
            torch.zeros(batch_size, module.n_head, 0, module.d_k, device=device),
            torch.tensor(0, dtype=torch.long, device=device)
        )
    elif isinstance(module, StreamingConformer):
        initial_states = module.get_initial_states(batch_size, device=device)
    elif isinstance(module, ConvSubsampling):
        initial_states = module.pad_buffer.expand(batch_size, -1, -1, -1).to(device)
    elif isinstance(module, ConformerConvModule):
        initial_states = module.cache_pad.expand(batch_size, -1, -1).to(device)
    else:
        # Default or other modules
        initial_states = None

    # Batch inference
    x = torch.rand(*input_shape).to(device)
    
    with torch.no_grad():
        if isinstance(module, CausalMultiHeadAttention):
            y_batch, _ = module(x, initial_states)
        elif isinstance(module, StreamingConformer):
            (y_batch, _, _), _ = module(x, initial_states)
        else:
            y_batch, _ = module(x, initial_states)
            
    # Streaming inference
    states = initial_states
    y_stream_list = []
    seq_len = input_shape[1]
    
    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            
            if isinstance(module, CausalMultiHeadAttention):
                y_chunk, states = module(chunk, states)
            elif isinstance(module, StreamingConformer):
                (y_chunk, _, _), states = module(chunk, states)
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

@pytest.mark.parametrize("chunk_size", [2, 8, 40, 80, 32])
def test_conv_subsampling_consistency(chunk_size):
    # Input: (B, T, F) -> (1, 400, 80)
    input_shape = (1, 400, config.N_BINS)
    subsampling = ConvSubsampling(config.N_BINS, config.D_MODEL)
    
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
    attn = CausalMultiHeadAttention(config.D_MODEL, config.N_HEAD)
    
    diff = check_streaming_consistency(attn, input_shape, chunk_size)
    assert diff < 1e-5, f"Attention consistency failed. Diff: {diff}"

@pytest.mark.parametrize("chunk_size", [40, 80, 32])
def test_model_consistency(chunk_size):
    input_shape = (1, 400, config.N_BINS)
    model = StreamingConformer(num_layers=4)
    
    diff = check_streaming_consistency(model, input_shape, chunk_size)
    # Increased tolerance slightly for deeper models with many LayerNorms
    assert diff < 2e-5, f"StreamingConformer consistency failed. Diff: {diff}"

def test_cache_limit_and_pe_consistency():
    """Test if cache limit works and PE index remains valid."""
    d_model = 144
    n_head = 4
    max_cache = config.MAX_CACHE_LEN
    
    model = StreamingConformer(d_model=d_model, n_head=n_head, num_layers=1)
    model.eval()
    
    # Process enough chunks to exceed MAX_CACHE_LEN + causal_mask margin (100)
    # With subsampling (factor 2), we need offset > 1100 (MAX_CACHE_LEN + 100)
    # offset = num_chunks * chunk_size / 2 > 1100
    # num_chunks > 1100 * 2 / chunk_size = 22
    chunk_size = 100
    num_chunks = 25  # Enough to exceed causal_mask bounds
    
    device = torch.device('cpu')
    states = model.get_initial_states(1, device=device)

    with torch.no_grad():
        for i in range(num_chunks):
            # Use positive inputs for PCEN
            x = torch.rand(1, chunk_size, config.N_BINS).to(device)
            (logits, _, _), states = model(x, states)
            
            # Check attn cache size of the first layer
            # states -> (pcen_state, sub_cache, layer_states)
            # layer_states[0] -> (attn_cache, conv_cache)
            # attn_cache -> (k, v, offset)
            k_cache = states[2][0][0][0]
            assert k_cache.size(2) <= max_cache
            
            # Check offset is correctly managed
            offset = states[2][0][0][2]
            print(f"Chunk {i}: offset={offset.item()}, max_cache={max_cache}")
            assert offset.item() <= max_cache, f"offset {offset.item()} exceeds max_cache {max_cache}"

def test_pcen_ema_vectorized_equivalence():
    """Verify that vectorized PCEN EMA is equivalent to recursive implementation."""
    from model import PCEN
    import config
    
    batch_size = 2
    seq_len = 50
    n_bins = config.N_BINS
    
    pcen = PCEN(n_bins)
    x = torch.rand(batch_size, seq_len, n_bins)
    s_init = torch.rand(batch_size, 1, n_bins)
    
    # 1. Recursive implementation (original logic)
    # s = torch.exp(self.log_s) in model.py
    s_val = torch.exp(pcen.log_s).view(1, 1, -1)
    s_rec = torch.zeros(batch_size, seq_len, n_bins)
    # This loop exactly reproduces the original recursive implementation in model.py
    curr_s = s_init
    s_list = []
    for t in range(seq_len):
        curr_s = (1 - s_val) * curr_s + s_val * x[:, t:t+1, :]
        s_list.append(curr_s)
    s_rec = torch.cat(s_list, dim=1)
        
    # 2. Vectorized implementation (current model.py logic)
    from model import _pcen_ema_loop
    with torch.no_grad():
        # _pcen_ema_loop is a module-level function
        s_vec, _ = _pcen_ema_loop(x, s_init, s_val)
        
    # Check equivalence
    diff = torch.abs(s_rec - s_vec).max().item()
    assert diff < 1e-6, f"Vectorized EMA differs from recursive. Max diff: {diff}"

def test_narrow_vs_slice_equivalence():
    """Verify that narrow and slice operations produce identical results."""
    
    # Test 1: CausalMultiHeadAttention cache trimming logic
    batch_size = 2
    n_head = 4
    d_k = 64
    cache_len = 150
    max_cache = 100
    
    k = torch.rand(batch_size, n_head, cache_len, d_k)
    
    # Slice method (old implementation)
    k_slice = k[:, :, -max_cache:, :]
    
    # Narrow method (new implementation)
    k_full_len = k.size(2)
    k_start = k_full_len - max_cache
    actual_start = max(0, k_start)
    actual_len = k_full_len - actual_start
    k_narrow = k.narrow(2, actual_start, actual_len)
    
    # Check equivalence
    diff = torch.abs(k_slice - k_narrow).max().item()
    assert diff < 1e-6, f"narrow and slice differ in attention cache. Diff: {diff}"
    
    # Test 2: ConformerConvModule cache saving logic
    batch_size = 2
    d_model = 256
    kernel_size = 31
    cache_len_target = kernel_size - 1
    total_len = 100
    
    x = torch.rand(batch_size, d_model, total_len)
    
    # Slice method (old implementation)
    x_slice = x[:, :, -cache_len_target:]
    
    # Narrow method (new implementation)
    start_idx = total_len - cache_len_target
    x_narrow = x.narrow(2, start_idx, cache_len_target)
    
    # Check equivalence
    diff = torch.abs(x_slice - x_narrow).max().item()
    assert diff < 1e-6, f"narrow and slice differ in conv cache. Diff: {diff}"

def test_pcen_ema_large_sequence():
    """Verify numerical stability of PCEN EMA for large sequence lengths."""
    from model import PCEN
    
    pcen = PCEN(config.N_BINS)
    
    for T in [100, 500, 1000]:
        x = torch.rand(1, T, config.N_BINS)
        state = torch.rand(1, 1, config.N_BINS)
        
        with torch.no_grad():
            E, new_state = pcen(x, state)
        
        # Check no NaN values
        assert not torch.isnan(E).any(), f"PCEN output contains NaN for T={T}"
        assert not torch.isnan(new_state).any(), f"PCEN new_state contains NaN for T={T}"
        
        # Check all values are finite
        assert torch.isfinite(E).all(), f"PCEN output contains inf for T={T}"
        assert torch.isfinite(new_state).all(), f"PCEN new_state contains inf for T={T}"

def test_get_initial_states_cache_len():
    """Verify get_initial_states handles different cache_len values correctly."""
    model = StreamingConformer(num_layers=2)
    
    for cache_len in [0, 10, 50]:
        states = model.get_initial_states(1, device="cpu", cache_len=cache_len)
        
        pcen_state, sub_cache, layer_states = states
        
        # Check pcen_state shape
        assert pcen_state.shape == (1, 1, config.N_BINS), f"pcen_state shape mismatch for cache_len={cache_len}"
        
        # Check sub_cache shape
        assert sub_cache.shape == (1, 1, 2, config.N_BINS), f"sub_cache shape mismatch for cache_len={cache_len}"
        
        # Check layer_states count
        assert len(layer_states) == 2, f"layer_states count mismatch for cache_len={cache_len}"
        
        # Check each layer's cache shapes
        for i, (attn_cache, conv_cache) in enumerate(layer_states):
            k, v, offset = attn_cache
            assert k.shape == (1, config.N_HEAD, cache_len, config.D_MODEL // config.N_HEAD), \
                f"Layer {i}: k shape mismatch for cache_len={cache_len}"
            assert v.shape == (1, config.N_HEAD, cache_len, config.D_MODEL // config.N_HEAD), \
                f"Layer {i}: v shape mismatch for cache_len={cache_len}"
            assert offset.item() == cache_len, \
                f"Layer {i}: offset mismatch for cache_len={cache_len}"
            assert conv_cache.shape == (1, config.D_MODEL, config.KERNEL_SIZE - 1), \
                f"Layer {i}: conv_cache shape mismatch for cache_len={cache_len}"

def test_conv_subsampling_boundary():
    """Verify ConvSubsampling handles boundary condition correctly."""
    subsampling = ConvSubsampling(config.N_BINS, config.D_MODEL)
    subsampling.eval()
    
    # Input length = 1 (after concatenating cache: l_in = 2 + 1 = 3, n_out = (3-3)//2+1 = 1)
    x = torch.rand(1, 1, config.N_BINS)
    cache = torch.zeros(1, 1, 2, config.N_BINS)
    
    with torch.no_grad():
        y, new_cache = subsampling(x, cache)
    
    # Expected output length = n_out = 1
    assert y.shape == (1, 1, config.D_MODEL), f"Expected shape (1, 1, {config.D_MODEL}), got {y.shape}"

def test_conv_subsampling_too_short():
    """Verify ConvSubsampling raises error when concatenated input length < 3."""
    subsampling = ConvSubsampling(config.N_BINS, config.D_MODEL)
    subsampling.eval()
    
    # Input length = 0 (after concatenating cache: l_in = 2 + 0 = 2 < 3, should raise error)
    x = torch.rand(1, 0, config.N_BINS)
    cache = torch.zeros(1, 1, 2, config.N_BINS)
    
        # In eager mode, torch._check should raise an error
    with pytest.raises(RuntimeError, match="Expected cond to be True, but got False"):
        with torch.no_grad():
            y, new_cache = subsampling(x, cache)

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [20, 40, 80, 160])
def test_dynamic_shape_inference(batch_size, seq_len):
    """Verify model inference works correctly with different batch sizes and sequence lengths."""
    model = StreamingConformer(num_layers=2)
    model.eval()
    
    states = model.get_initial_states(batch_size, device="cpu")
    x = torch.rand(batch_size, seq_len, config.N_BINS)
    
    with torch.no_grad():
        (logits, sig_out, bound_out), new_states = model(x, states)
    
    # Check output shapes
    expected_seq_len = seq_len // config.SUBSAMPLING_RATE
    assert logits.shape == (batch_size, expected_seq_len, config.NUM_CLASSES), \
        f"logits shape mismatch: expected ({batch_size}, {expected_seq_len}, {config.NUM_CLASSES}), got {logits.shape}"
    assert sig_out.shape == (batch_size, expected_seq_len, config.NUM_SIGNAL_CLASSES), \
        f"signal logits shape mismatch: expected ({batch_size}, {expected_seq_len}, {config.NUM_SIGNAL_CLASSES}), got {sig_out.shape}"
    assert bound_out.shape == (batch_size, expected_seq_len, 1), \
        f"boundary logits shape mismatch: expected ({batch_size}, {expected_seq_len}, 1), got {bound_out.shape}"