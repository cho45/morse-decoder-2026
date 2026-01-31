import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from stream_decode import CTCDecoder

def test_stream_decode_ctc_space():
    decoder = CTCDecoder(config.ID_TO_CHAR)
    space_id = config.CHAR_TO_ID[' ']
    c_id = config.CHAR_TO_ID['C']
    
    # Chunk 1: "C"
    logits1 = torch.full((1, 5, config.NUM_CLASSES), -10.0)
    logits1[0, 2, c_id] = 10.0
    sig_logits1 = torch.zeros((1, 5, 4))
    bound_logits1 = torch.zeros((1, 5, 1))
    
    res1 = decoder.decode(logits1, sig_logits1, bound_logits1)
    assert res1 == "C"
    
    # Chunk 2: " " (CTC Space)
    logits2 = torch.full((1, 5, config.NUM_CLASSES), -10.0)
    logits2[0, 2, space_id] = 10.0
    sig_logits2 = torch.zeros((1, 5, 4))
    bound_logits2 = torch.zeros((1, 5, 1))
    
    res2 = decoder.decode(logits2, sig_logits2, bound_logits2)
    assert res2 == " "

def test_stream_decode_no_duplicate_space():
    decoder = CTCDecoder(config.ID_TO_CHAR)
    space_id = config.CHAR_TO_ID[' ']
    c_id = config.CHAR_TO_ID['C']
    
    # Chunk 1: "C" + Signal Head Space detection
    logits1 = torch.full((1, 5, config.NUM_CLASSES), -10.0)
    logits1[0, 2, c_id] = 10.0
    sig_logits1 = torch.zeros((1, 5, 4))
    sig_logits1[0, 4, 3] = 10.0 # Word space (should be ignored)
    bound_logits1 = torch.zeros((1, 5, 1))
    
    res1 = decoder.decode(logits1, sig_logits1, bound_logits1)
    # Signal Head is ignored
    assert res1 == "C"
    
    # Chunk 2: " " (CTC Space)
    logits2 = torch.full((1, 5, config.NUM_CLASSES), -10.0)
    logits2[0, 2, space_id] = 10.0
    sig_logits2 = torch.zeros((1, 5, 4))
    bound_logits2 = torch.zeros((1, 5, 1))
    
    res2 = decoder.decode(logits2, sig_logits2, bound_logits2)
    # CTC space should be decoded directly
    assert res2 == " "