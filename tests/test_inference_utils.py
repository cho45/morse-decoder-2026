import pytest
import torch
import numpy as np
from inference_utils import calculate_cer, map_prosigns, unmap_prosigns, decode_multi_task
import config

def test_prosign_mapping():
    text = "CQ DE K <BT> TEST <AR>"
    mapped = map_prosigns(text)
    # Prosigns should be single characters
    assert "<BT>" not in mapped
    assert "<AR>" not in mapped
    
    unmapped = unmap_prosigns(mapped)
    assert unmapped == text

def test_calculate_cer():
    # Exact match
    assert calculate_cer("ABC", "ABC") == 0.0
    # Substitution
    assert calculate_cer("ABC", "ADC") == 1/3
    # Deletion
    assert calculate_cer("ABC", "AB") == 1/3
    # Insertion
    assert calculate_cer("ABC", "ABCD") == 1/3
    # Prosign match (should count as 1 char)
    assert calculate_cer("<BT> K", "<BT> K") == 0.0
    # Prosign mismatch
    assert calculate_cer("<BT> K", "<AR> K") == 0.5 # <BT> replaced by <AR> (1 char)
    # Space should be ignored in CER
    assert calculate_cer("A B C", "ABC") == 0.0

def test_decode_multi_task_basic():
    # Setup dummy logits (Batch=1, Time=10, Classes)
    id_e = config.CHAR_TO_ID['E']
    id_t = config.CHAR_TO_ID['T']
    ctc_logits = torch.zeros(1, 10, config.NUM_CLASSES)
    ctc_logits[0, 2, id_e] = 10.0 # 'E' at t=2
    ctc_logits[0, 5, id_t] = 10.0 # 'T' at t=5
    
    # Signal: 0:bg, 1:dit, 2:dah, 3:word
    sig_logits = torch.zeros(1, 10, 4)
    sig_logits[0, :, 0] = 5.0 # default background
    
    # Boundary
    bound_probs = torch.zeros(1, 10, 1)
    
    # Simple decode (no spaces)
    decoded, timed = decode_multi_task(ctc_logits[0], sig_logits[0], bound_probs[0])
    assert decoded == "ET"
    assert timed == [('E', 2), ('T', 5)]

def test_decode_multi_task_with_space():
    id_e = config.CHAR_TO_ID['E']
    id_t = config.CHAR_TO_ID['T']
    ctc_logits = torch.zeros(1, 10, config.NUM_CLASSES)
    ctc_logits[0, 1, id_e] = 10.0 # 'E'
    ctc_logits[0, 8, id_t] = 10.0 # 'T'
    
    sig_logits = torch.zeros(1, 10, 4)
    sig_logits[0, :, 0] = 2.0
    sig_logits[0, 4:6, 3] = 10.0 # Word space at t=4,5
    
    bound_probs = torch.zeros(1, 10, 1)
    bound_probs[0, 4:6, 0] = 0.8 # Boundary Head also confirms space region
    
    decoded, _ = decode_multi_task(ctc_logits[0], sig_logits[0], bound_probs[0])
    assert decoded == "E T"

def test_decode_multi_task_space_gating():
    # Space predicted by Signal Head but NOT confirmed by Boundary Head
    id_e = config.CHAR_TO_ID['E']
    id_t = config.CHAR_TO_ID['T']
    ctc_logits = torch.zeros(1, 10, config.NUM_CLASSES)
    ctc_logits[0, 1, id_e] = 10.0 # 'E'
    ctc_logits[0, 8, id_t] = 10.0 # 'T'

    sig_logits = torch.zeros(1, 10, 4)
    sig_logits[0, 4:6, 3] = 10.0 # Word space

    bound_probs = torch.zeros(1, 10, 1) # All zero (no boundary)

    decoded, _ = decode_multi_task(ctc_logits[0], sig_logits[0], bound_probs[0])
    assert decoded == "ET" # Space should be gated out

def test_decode_multi_task_space_timing():
    """
    Test space insertion timing with realistic CW signal timeline.

    Timeline:
    - t=2: 'H' fires (CTC) after H signal ends
    - t=4: 'I' fires (CTC) after I signal ends
    - t=7-9: word space signal detected (sig_logits)
    - t=12: 'T' fires (CTC) after T signal ends

    Expected: "HI T" where space is inserted after 'I', not before 'T'.
    This reflects that the space logically belongs to the word "HI", not to "T".
    """
    id_h = config.CHAR_TO_ID['H']
    id_i = config.CHAR_TO_ID['I']
    id_t = config.CHAR_TO_ID['T']

    ctc_logits = torch.zeros(1, 15, config.NUM_CLASSES)
    ctc_logits[0, 2, id_h] = 10.0  # 'H' at t=2
    ctc_logits[0, 4, id_i] = 10.0  # 'I' at t=4
    ctc_logits[0, 12, id_t] = 10.0 # 'T' at t=12

    sig_logits = torch.zeros(1, 15, 4)
    sig_logits[0, :, 0] = 2.0  # background
    sig_logits[0, 7:10, 3] = 10.0  # Word space at t=7-9

    bound_probs = torch.zeros(1, 15, 1)
    bound_probs[0, 7:10, 0] = 0.8  # Boundary confirms space region

    decoded, timed = decode_multi_task(ctc_logits[0], sig_logits[0], bound_probs[0])

    # Space should be inserted after 'I', creating "HI T"
    assert decoded == "HI T"

    # timed_output should only contain characters (not spaces)
    assert timed == [('H', 2), ('I', 4), ('T', 12)]

def test_decode_multi_task_multiple_words():
    """
    Test with multiple words: "CQ DE K"

    Timeline:
    - t=1: 'C', t=3: 'Q' -> space at t=5-6
    - t=8: 'D', t=10: 'E' -> space at t=12-13
    - t=15: 'K'
    """
    id_c = config.CHAR_TO_ID['C']
    id_q = config.CHAR_TO_ID['Q']
    id_d = config.CHAR_TO_ID['D']
    id_e = config.CHAR_TO_ID['E']
    id_k = config.CHAR_TO_ID['K']

    ctc_logits = torch.zeros(1, 20, config.NUM_CLASSES)
    ctc_logits[0, 1, id_c] = 10.0
    ctc_logits[0, 3, id_q] = 10.0
    ctc_logits[0, 8, id_d] = 10.0
    ctc_logits[0, 10, id_e] = 10.0
    ctc_logits[0, 15, id_k] = 10.0

    sig_logits = torch.zeros(1, 20, 4)
    sig_logits[0, :, 0] = 2.0
    sig_logits[0, 5:7, 3] = 10.0   # Space after CQ
    sig_logits[0, 12:14, 3] = 10.0 # Space after DE

    bound_probs = torch.zeros(1, 20, 1)
    bound_probs[0, 5:7, 0] = 0.8
    bound_probs[0, 12:14, 0] = 0.8

    decoded, timed = decode_multi_task(ctc_logits[0], sig_logits[0], bound_probs[0])

    assert decoded == "CQ DE K"
    assert timed == [('C', 1), ('Q', 3), ('D', 8), ('E', 10), ('K', 15)]