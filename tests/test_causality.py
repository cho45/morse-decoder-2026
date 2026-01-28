import pytest
import torch
import config
import numpy as np
import random
from model import StreamingConformer, ConvSubsampling

def test_subsampling_causality():
    """
    Ensure that future input frames do not affect past output frames in ConvSubsampling.
    """
    subsampling = ConvSubsampling(config.N_BINS, config.D_MODEL)
    subsampling.eval()
    
    T = 20
    # Use positive inputs for PCEN
    x1 = torch.rand(1, T, config.N_BINS)
    x2 = x1.clone()
    # Modify the future frame (e.g., frame 10)
    x2[:, 10:, :] += 1.0
    
    with torch.no_grad():
        y1, _ = subsampling(x1)
        y2, _ = subsampling(x2)
    
    # Output at frame i corresponds to input around 2*i.
    # If causality holds, output frame i should be identical if 2*i < 10.
    # With kernel=3, stride=2, output 4 uses inputs [7, 8, 9] (if causal)
    # or [8, 9, 10] (if non-causal with symmetric padding).
    
    # Check output frames 0 to 4
    diff = torch.abs(y1[:, :5, :] - y2[:, :5, :]).max().item()
    assert diff < 1e-6, f"Causality violation in ConvSubsampling: future frames affected past output. Diff: {diff}"

def test_model_causality():
    """
    Ensure the entire StreamingConformer is causal.
    """
    # Set seed for reproducibility and avoid extreme random values
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = StreamingConformer(num_layers=2)
    model.eval()
    
    T = 40
    # Use positive inputs for PCEN
    x1 = torch.rand(1, T, config.N_BINS)
    x2 = x1.clone()
    x2[:, 20:, :] += 0.5
    
    with torch.no_grad():
        (y1, _, _), _ = model(x1)
        (y2, _, _), _ = model(x2)
    
    # Check output frames before the change (approx T/2)
    diff = torch.abs(y1[:, :10, :] - y2[:, :10, :]).max().item()
    assert diff < 1e-6, f"Causality violation in StreamingConformer. Diff: {diff}"

def test_strict_streaming_consistency():
    """
    Streaming output must match batch output EXACTLY for any chunk size.
    """
    torch.manual_seed(42)
    model = StreamingConformer(num_layers=2)
    model.eval()
    
    T = 40
    # Use positive inputs for PCEN
    x = torch.rand(1, T, config.N_BINS)
    
    with torch.no_grad():
        (y_batch, _, _), _ = model(x)
    
    # Test different chunk sizes
    for chunk_size in [2, 4, 10]:
        y_streams = []
        states = None
        with torch.no_grad():
            for i in range(0, T, chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                (y_chunk, _, _), states = model(chunk, states)
                y_streams.append(y_chunk)
        
        y_stream = torch.cat(y_streams, dim=1)
        
        # We check the minimum length to handle trailing frames in subsampling
        min_len = min(y_batch.size(1), y_stream.size(1))
        diff = torch.abs(y_batch[:, :min_len, :] - y_stream[:, :min_len, :]).max().item()
        assert diff < 1e-5, f"Streaming inconsistency for chunk_size {chunk_size}. Diff: {diff}"