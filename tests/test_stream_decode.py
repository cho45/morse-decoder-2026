import pytest
import torch
import numpy as np
import os
import tempfile
import argparse
from stream_decode import CTCDecoder, StreamDecoder
from model import StreamingConformer
import config

def test_ctc_decoder_greedy():
    id_to_char = {1: 'A', 2: 'B', 3: 'C'}
    decoder = CTCDecoder(id_to_char)
    
    # Simple sequence: [blank, A, A, blank, B, blank, C, C]
    # Should decode to "ABC"
    # logits shape: (1, T, C)
    logits = torch.zeros(1, 8, 4)
    logits[0, 0, 0] = 1.0 # blank
    logits[0, 1, 1] = 1.0 # A
    logits[0, 2, 1] = 1.0 # A
    logits[0, 3, 0] = 1.0 # blank
    logits[0, 4, 2] = 1.0 # B
    logits[0, 5, 0] = 1.0 # blank
    logits[0, 6, 3] = 1.0 # C
    logits[0, 7, 3] = 1.0 # C
    
    # We need to reset last_id for independent tests if needed,
    # but here we just create a new decoder.
    # Dummy sig_logits (all zeros, no spaces)
    sig_logits = torch.zeros(1, 8, 6)
    # Dummy boundary_logits (all ones, always allow)
    boundary_logits = torch.ones(1, 8, 1) * 10.0
    decoded = decoder.decode(logits, sig_logits, boundary_logits)
    assert decoded == "ABC"

def test_ctc_decoder_stateful():
    id_to_char = {1: 'A', 2: 'B'}
    decoder = CTCDecoder(id_to_char)
    
    # First chunk ends with 'A'
    logits1 = torch.zeros(1, 2, 3)
    logits1[0, 0, 1] = 1.0 # A
    logits1[0, 1, 1] = 1.0 # A
    sig_logits1 = torch.zeros(1, 2, 6)
    boundary_logits1 = torch.ones(1, 2, 1) * 10.0
    assert decoder.decode(logits1, sig_logits1, boundary_logits1) == "A"
    assert decoder.last_id == 1
    
    # Second chunk starts with 'A', then 'B'
    logits2 = torch.zeros(1, 2, 3)
    logits2[0, 0, 1] = 1.0 # A (should be ignored as it is same as last_id)
    logits2[0, 1, 2] = 1.0 # B
    sig_logits2 = torch.zeros(1, 2, 6)
    boundary_logits2 = torch.ones(1, 2, 1) * 10.0
    assert decoder.decode(logits2, sig_logits2, boundary_logits2) == "B"
    assert decoder.last_id == 2

@pytest.fixture
def dummy_checkpoint():
    # Create a dummy checkpoint for StreamDecoder
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        device = "cpu"
        model = StreamingConformer(
            n_mels=config.N_BINS,
            num_classes=config.NUM_CLASSES,
            d_model=config.D_MODEL,
            n_head=config.N_HEAD,
            num_layers=config.NUM_LAYERS,
        )
        args = argparse.Namespace(
            lr=0.001,
            weight_decay=0.0001,
            samples_per_epoch=100,
            batch_size=16,
            save_dir='checkpoints'
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args
        }, f.name)
        return f.name

def test_stream_decoder_init(dummy_checkpoint):
    decoder = StreamDecoder(dummy_checkpoint, device="cpu")
    assert decoder.model is not None
    # states is (pcen_state, sub_cache, layer_states)
    assert decoder.states is not None
    assert len(decoder.states) == 3
    # Ensure states are initialized as tensors
    assert torch.is_tensor(decoder.states[0])
    assert torch.is_tensor(decoder.states[1])
    os.remove(dummy_checkpoint)

def test_stream_decoder_process_chunk(dummy_checkpoint):
    decoder = StreamDecoder(dummy_checkpoint, device="cpu")
    
    # To trigger processing, we need len(combined) >= samples_needed
    chunk_size = 20000
    audio_chunk = np.random.randn(chunk_size).astype(np.float32)
    
    # Process chunk
    decoder.process_chunk(audio_chunk)
    
    # Check if states are updated
    assert decoder.states is not None
    # states is (pcen_state, sub_cache, layer_states)
    # layer_states is a list of (attn_cache, conv_cache) for each layer
    assert len(decoder.states) == 3
    assert len(decoder.states[2]) == config.NUM_LAYERS
    
    os.remove(dummy_checkpoint)