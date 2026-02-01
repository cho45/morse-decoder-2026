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
    assert timing[0] == (1, pytest.approx(1.2 / 20, rel=1e-2)) # dot (Dit=1)
    assert timing[1] == (3, pytest.approx(1.2 / 20, rel=1e-2)) # inter-symbol (Intra-char-space=3)
    assert timing[2] == (2, pytest.approx(3 * 1.2 / 20, rel=1e-2)) # dash (Dah=2)

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
    for class_id, duration in timing:
        if class_id == 5 and abs(duration - word_space_len) < 1e-4: # Inter-word space = 5
            has_word_space = True
    assert has_word_space

def test_waveform_generation():
    gen = MorseGenerator(sample_rate=config.SAMPLE_RATE)
    timing = [(1, 0.1), (3, 0.1), (2, 0.3)]
    waveform, signal_labels, boundary_labels, _ = gen.generate_waveform(timing, frequency=700)
    
    assert isinstance(waveform, np.ndarray)
    assert isinstance(signal_labels, np.ndarray)
    assert isinstance(boundary_labels, np.ndarray)
    assert len(waveform) > 0
    assert np.max(np.abs(waveform)) <= 1.0

def test_hf_simulator():
    sim = HFChannelSimulator(sample_rate=config.SAMPLE_RATE)
    waveform = np.sin(2 * np.pi * 700 * np.arange(config.SAMPLE_RATE) / config.SAMPLE_RATE)
    
    faded = sim.apply_fading(waveform)
    assert faded.shape == waveform.shape
    assert not np.array_equal(faded, waveform)
    
    noised = sim.apply_noise(waveform, snr_2500=10)
    assert noised.shape == waveform.shape
    
    qrm = sim.apply_qrm(waveform)
    assert qrm.shape == waveform.shape
    
    filtered = sim.apply_filter(waveform)
    assert filtered.shape == waveform.shape

def test_generate_sample():
    waveform, label, signal_labels, boundary_labels = generate_sample("TEST", wpm=20, snr_2500=20)
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert isinstance(signal_labels, torch.Tensor)
    assert isinstance(boundary_labels, torch.Tensor)
    assert label == "TEST "
    assert waveform.ndim == 1
    assert isinstance(signal_labels, torch.Tensor)

def test_cw_dataset():
    num_samples = 5
    dataset = CWDataset(num_samples=num_samples)
    assert len(dataset) == num_samples
    
    # Updated to expect (waveform, label, wpm, signal_labels, boundary_labels, is_phrase)
    item = dataset[0]
    assert len(item) == 6
    waveform, label, wpm, signal_labels, boundary_labels, is_phrase = item
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(label, str)
    assert isinstance(wpm, int)
    assert len(label) > 0

def test_data_density():
    """Verify that the 10s window is used efficiently at high WPM."""
    # Test with high WPM to see if we fill the 10s window
    dataset = CWDataset(num_samples=20, min_wpm=40, max_wpm=40, phrase_prob=0.5)
    densities = []
    for i in range(len(dataset)):
        waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[i]
        # signal_labels: 1: Dit, 2: Dah
        sig_indices = torch.where((signal_labels == 1) | (signal_labels == 2))[0]
        if len(sig_indices) > 0:
            duration_frames = sig_indices[-1].item() - sig_indices[0].item()
            densities.append(duration_frames / len(signal_labels))
        else:
            densities.append(0.0)
    
    avg_density = np.mean(densities)
    print(f"Average density at 40 WPM: {avg_density:.2f}")
    # With max_len=10 limit restored, density should be relatively low (around 0.3)
    assert avg_density < 0.5, f"Average density {avg_density:.2f} is too high, max_len might not be working"

def test_strict_max_len():
    """Verify that the token count never exceeds max_len, including spaces."""
    max_len = 15
    dataset = CWDataset(num_samples=50, min_wpm=40, max_wpm=40, max_len=max_len, phrase_prob=0.5)
    gen = MorseGenerator()
    
    for i in range(len(dataset)):
        waveform, label, wpm, signal_labels, boundary_labels, is_phrase = dataset[i]
        tokens = gen.text_to_morse_tokens(label)
        # label は末尾に " " が付く仕様なので、実質的なトークン数は tokens の長さ
        token_count = len(tokens)
        assert token_count <= max_len, f"Token count {token_count} exceeds max_len {max_len} (Phrase: {is_phrase}, Text: '{label}')"

def test_data_no_overflow():
    """Verify that the signal never overflows the max_duration (10s)."""
    # Use very low WPM and long text to try and force overflow
    gen = MorseGenerator(sample_rate=config.SAMPLE_RATE)
    text = "CQ CQ CQ DE KILO CODE " * 5
    wpm = 10
    max_duration = 10.0
    
    # This should handle overflow gracefully (truncate or raise error, but here we expect safety)
    timing = gen.generate_timing(text, wpm=wpm)
    waveform, signal_labels, boundary_labels, _ = gen.generate_waveform(timing, wpm=wpm, max_duration=max_duration)
    
    # Check if the waveform length is exactly max_duration
    assert len(waveform) == int(max_duration * config.SAMPLE_RATE)
    
    # Check if the last frame is NOT a signal (should have some safety margin)
    # 0: Background/Space
    assert signal_labels[-1] == 0, "Signal overflowed at the end of the window"
    
    # Check timing sum
    total_timing = sum(t[1] for t in timing)
    # Even if timing is long, generate_waveform should have truncated or handled it
    # We'll check if the resulting labels are consistent with max_duration
    assert len(signal_labels) == (len(waveform) - config.N_FFT) // config.HOP_LENGTH + 1
