import torch
from model import StreamingConformer
import config

def test_model_multi_output():
    batch_size = 2
    seq_len = 100
    n_mels = config.N_BINS
    num_classes = config.NUM_CLASSES
    
    model = StreamingConformer(n_mels=n_mels, num_classes=num_classes)
    x = torch.randn(batch_size, seq_len, n_mels)
    
    states = model.get_initial_states(batch_size, x.device)
    (logits, signal_logits, boundary_logits), _ = model(x, states)
    
    # CTC Logits
    assert logits.shape == (batch_size, seq_len // config.SUBSAMPLING_RATE, num_classes)
    
    # Signal Detection Logits (Multi-class: config.NUM_SIGNAL_CLASSES)
    assert signal_logits.shape == (batch_size, seq_len // config.SUBSAMPLING_RATE, config.NUM_SIGNAL_CLASSES)
    assert signal_logits.dtype == torch.float32

def test_streaming_multi_output():
    model = StreamingConformer()
    model.eval()
    
    chunk_size = 40
    x = torch.randn(1, chunk_size, config.N_BINS)
    
    with torch.no_grad():
        states = model.get_initial_states(1, x.device)
        (logits, signal_logits, boundary_logits), states = model(x, states)
        
        assert logits.shape == (1, chunk_size // config.SUBSAMPLING_RATE, config.NUM_CLASSES)
        assert signal_logits.shape == (1, chunk_size // config.SUBSAMPLING_RATE, config.NUM_SIGNAL_CLASSES)
        assert states is not None