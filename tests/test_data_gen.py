import pytest
import torch
import numpy as np
from data_gen import MorseGenerator, HFChannelSimulator, CWDataset, generate_sample
import config

def test_morse_timing():
    gen = MorseGenerator(sample_rate=1000)
    # "A" is ".-"
    # dot: 1 unit, inter-symbol: 1 unit, dash: 3 units
    timing = gen.generate_timing("A", wpm=20, jitter=0)
    
    assert len(timing) == 3 # dot, inter-symbol, dash
    assert timing[0] == (True, pytest.approx(1.2 / 20, rel=1e-2)) # dot
    assert timing[1] == (False, pytest.approx(1.2 / 20, rel=1e-2)) # inter-symbol
    assert timing[2] == (True, pytest.approx(3 * 1.2 / 20, rel=1e-2)) # dash

def test_morse_timing_spaces():
    gen = MorseGenerator(sample_rate=1000)
    # "A B" 
    # A: . (1) space (1) - (3)
    # Inter-character space: 3 units
    # B: - (3) space (1) . (1) space (1) . (1) space (1) . (1)
    # Word space: 7 units
    timing = gen.generate_timing("A B", wpm=20, farnsworth_wpm=20, jitter=0)
    
    # Text "A B" split by space gives words ["A", "B"]
    # Word "A" has char "A" -> symbols ".", "-"
    # Word "B" has char "B" -> symbols "-", ".", ".", "."
    
    # Find word space
    has_word_space = False
    dot_len = 1.2 / 20
    word_space_len = 7 * dot_len
    for is_on, duration in timing:
        if not is_on and abs(duration - word_space_len) < 1e-4:
            has_word_space = True
    assert has_word_space

def test_waveform_generation():
    gen = MorseGenerator(sample_rate=config.SAMPLE_RATE)
    timing = [(True, 0.1), (False, 0.1), (True, 0.3)]
    waveform = gen.generate_waveform(timing, frequency=700)
    
    assert isinstance(waveform, np.ndarray)
    assert len(waveform) > 0
    assert np.max(np.abs(waveform)) <= 1.0

def test_hf_simulator():
    sim = HFChannelSimulator(sample_rate=config.SAMPLE_RATE)
    waveform = np.sin(2 * np.pi * 700 * np.arange(config.SAMPLE_RATE) / config.SAMPLE_RATE)
    
    faded = sim.apply_fading(waveform)
    assert faded.shape == waveform.shape
    assert not np.array_equal(faded, waveform)
    
    noised = sim.apply_noise(waveform, snr_db=10)
    assert noised.shape == waveform.shape
    
    qrm = sim.apply_qrm(waveform)
    assert qrm.shape == waveform.shape
    
    filtered = sim.apply_filter(waveform)
    assert filtered.shape == waveform.shape

def test_generate_sample():
    waveform, label = generate_sample("TEST", wpm=20, snr_db=20)
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert label == "TEST"
    assert waveform.ndim == 1

def test_cw_dataset():
    num_samples = 5
    dataset = CWDataset(num_samples=num_samples)
    assert len(dataset) == num_samples
    
    waveform, label = dataset[0]
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert len(label) > 0