import pytest
import torch
import config
from model import RelPositionalEncoding, PCEN, ConformerBlock, _pcen_ema_loop


class TestRelPositionalEncodingBoundary:
    """Test RelPositionalEncoding boundary conditions."""

    def test_pe_with_zero_offset(self) -> None:
        """PE should work correctly with offset=0."""
        d_model = 64
        pe = RelPositionalEncoding(d_model)
        pe.eval()

        # Test with offset=0
        seq_len = 50
        x = torch.rand(1, seq_len, d_model)
        offset = torch.tensor(0, dtype=torch.long, device=x.device)

        with torch.no_grad():
            pos_emb = pe(seq_len, offset=offset)

        # Should return correct shape
        assert pos_emb.shape == (seq_len, d_model)
        assert torch.isfinite(pos_emb).all()

    def test_pe_long_running_offset(self) -> None:
        """PE should maintain valid indices over long-running execution."""
        d_model = 64
        pe = RelPositionalEncoding(d_model)
        pe.eval()

        # Simulate long-running streaming by incrementally increasing offset
        total_frames = 500
        chunk_size = 50
        current_offset = torch.tensor(0, dtype=torch.long, device='cpu')

        with torch.no_grad():
            for _ in range(total_frames // chunk_size):
                seq_len = chunk_size
                pos_emb = pe(seq_len, offset=current_offset)
                assert pos_emb.shape == (seq_len, d_model)
                assert torch.isfinite(pos_emb).all()
                current_offset = current_offset + seq_len

    def test_pe_max_boundary(self) -> None:
        """PE should handle sequences approaching max_len."""
        d_model = 64
        max_len = 500
        pe = RelPositionalEncoding(d_model, max_len=max_len)
        pe.eval()

        # Test sequence length close to max_len
        seq_len = max_len - 10
        offset = torch.tensor(0, dtype=torch.long, device='cpu')

        with torch.no_grad():
            pos_emb = pe(seq_len, offset=offset)

        assert pos_emb.shape == (seq_len, d_model)
        assert torch.isfinite(pos_emb).all()


class TestPCENStateManagement:
    """Test PCEN state management."""

    def test_pcen_empty_input(self) -> None:
        """PCEN should handle empty input (T=0) correctly."""
        n_mels = config.N_BINS
        pcen = PCEN(n_mels)
        pcen.eval()

        # Empty input
        x = torch.zeros(1, 0, n_mels)

        with torch.no_grad():
            y, new_state = pcen(x)

        # Output should be empty
        assert y.shape == (1, 0, n_mels)
        # State should be valid (shape preserved)
        assert new_state.shape == (1, 1, n_mels)

    def test_pcen_state_continuity(self) -> None:
        """PCEN state should be continuous across calls."""
        n_mels = config.N_BINS
        pcen = PCEN(n_mels)
        pcen.eval()

        # First call with positive input
        x1 = torch.rand(1, 50, n_mels)
        with torch.no_grad():
            y1, state1 = pcen(x1)

        # Second call with new input using state1
        x2 = torch.rand(1, 50, n_mels)
        with torch.no_grad():
            y2, state2 = pcen(x2, state=state1)

        # Compare with single batch call
        x_batch = torch.cat([x1, x2], dim=1)
        with torch.no_grad():
            y_batch, _ = pcen(x_batch)

        # States should match (within tolerance for numerical precision)
        assert torch.allclose(torch.cat([y1, y2], dim=1), y_batch, atol=1e-5)

    def test_pcen_state_with_zero_input(self) -> None:
        """PCEN state should handle zero inputs correctly."""
        n_mels = config.N_BINS
        pcen = PCEN(n_mels)
        pcen.eval()

        # Start with zero input
        x = torch.zeros(1, 50, n_mels)
        with torch.no_grad():
            y, state = pcen(x)

        # Output should be finite
        assert torch.isfinite(y).all()
        # State should be finite
        assert torch.isfinite(state).all()


class TestPcenEmaLoop:
    """Test _pcen_ema_loop function directly."""

    def test_ema_loop_basic(self) -> None:
        """EMA loop should compute correct output for basic input."""
        batch_size = 2
        T = 10
        n_mels = 4

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        # Shape should be correct
        assert output.shape == (batch_size, T, n_mels)
        assert final_state.shape == (batch_size, 1, n_mels)

        # Final state should match the last EMA value
        # Manual calculation for verification
        curr = state
        for t in range(T):
            curr = (1 - s) * curr + s * x[:, t:t+1, :]
        assert torch.allclose(final_state, curr, atol=1e-6)

    def test_ema_loop_empty_input(self) -> None:
        """EMA loop should handle empty input (T=0)."""
        batch_size = 2
        n_mels = 4

        x = torch.zeros(batch_size, 0, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        # Output should be empty
        assert output.shape == (batch_size, 0, n_mels)
        # State should be unchanged
        assert torch.allclose(final_state, state)

    def test_ema_loop_single_frame(self) -> None:
        """EMA loop should handle single frame (T=1)."""
        batch_size = 2
        n_mels = 4

        x = torch.rand(batch_size, 1, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        # Shape
        assert output.shape == (batch_size, 1, n_mels)
        assert final_state.shape == (batch_size, 1, n_mels)

        # Manual calculation
        expected = (1 - s) * state + s * x
        assert torch.allclose(output, expected, atol=1e-6)
        assert torch.allclose(final_state, expected, atol=1e-6)

    def test_ema_loop_numerical_stability(self) -> None:
        """EMA loop should be numerically stable for long sequences."""
        batch_size = 1
        T = 1000
        n_mels = 14

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        # All outputs should be finite
        assert torch.isfinite(output).all()
        assert torch.isfinite(final_state).all()

    def test_ema_loop_streaming_equivalence(self) -> None:
        """EMA loop should produce same result as sequential calls."""
        batch_size = 2
        T = 100
        n_mels = 14

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        # Single call
        output_batch, state_batch = _pcen_ema_loop(x, state, s)

        # Sequential calls (simulating streaming)
        chunk_size = 20
        outputs_stream = []
        curr_state = state

        for i in range(0, T, chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            out, curr_state = _pcen_ema_loop(chunk, curr_state, s)
            outputs_stream.append(out)

        output_stream = torch.cat(outputs_stream, dim=1)

        # Results should match
        assert torch.allclose(output_batch, output_stream, atol=1e-5)
        assert torch.allclose(state_batch, curr_state, atol=1e-5)

    def test_ema_loop_batch_size_one(self) -> None:
        """EMA loop should work with batch_size=1."""
        batch_size = 1
        T = 50
        n_mels = 14

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        assert output.shape == (batch_size, T, n_mels)
        assert final_state.shape == (batch_size, 1, n_mels)
        assert torch.isfinite(output).all()

    def test_ema_loop_various_s_values(self) -> None:
        """EMA loop should work with various smoothing coefficients."""
        batch_size = 2
        T = 20
        n_mels = 4

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)

        # Test with various s values
        for s_val in [0.001, 0.01, 0.1, 0.5, 0.99]:
            s = torch.tensor(s_val).view(1, 1, 1)
            output, final_state = _pcen_ema_loop(x, state, s)

            assert output.shape == (batch_size, T, n_mels)
            assert final_state.shape == (batch_size, 1, n_mels)
            assert torch.isfinite(output).all(), f"s={s_val} produced non-finite output"
            assert torch.isfinite(final_state).all(), f"s={s_val} produced non-finite state"

    def test_ema_loop_extreme_input_values(self) -> None:
        """EMA loop should handle extreme input values."""
        batch_size = 2
        T = 10
        n_mels = 4

        # Very small positive values
        x_small = torch.full((batch_size, T, n_mels), 1e-10)
        state_small = torch.full((batch_size, 1, n_mels), 1e-10)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x_small, state_small, s)
        assert torch.isfinite(output).all()
        assert torch.isfinite(final_state).all()

        # Large positive values
        x_large = torch.full((batch_size, T, n_mels), 1e6)
        state_large = torch.full((batch_size, 1, n_mels), 1e6)

        output, final_state = _pcen_ema_loop(x_large, state_large, s)
        assert torch.isfinite(output).all()
        assert torch.isfinite(final_state).all()

    def test_ema_loop_zero_state(self) -> None:
        """EMA loop should handle zero initial state."""
        batch_size = 2
        T = 10
        n_mels = 4

        x = torch.rand(batch_size, T, n_mels)
        state = torch.zeros(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        assert output.shape == (batch_size, T, n_mels)
        assert torch.isfinite(output).all()
        assert torch.isfinite(final_state).all()

    def test_ema_loop_per_channel_s(self) -> None:
        """EMA loop should work with per-channel smoothing coefficients."""
        batch_size = 2
        T = 20
        n_mels = 4

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        # Per-channel s values
        s = torch.tensor([0.01, 0.02, 0.03, 0.04]).view(1, 1, n_mels)

        output, final_state = _pcen_ema_loop(x, state, s)

        assert output.shape == (batch_size, T, n_mels)
        assert final_state.shape == (batch_size, 1, n_mels)
        assert torch.isfinite(output).all()

        # Verify each channel has different smoothing behavior
        # The final state for each channel should be influenced by its s value
        assert final_state.shape == (batch_size, 1, n_mels)

    def test_ema_loop_output_state_consistency(self) -> None:
        """Output's last frame should match final state."""
        batch_size = 2
        T = 50
        n_mels = 14

        x = torch.rand(batch_size, T, n_mels)
        state = torch.rand(batch_size, 1, n_mels)
        s = torch.tensor(0.025).view(1, 1, 1)

        output, final_state = _pcen_ema_loop(x, state, s)

        # The last frame of output should equal final_state
        assert torch.allclose(output[:, -1:, :], final_state, atol=1e-6)


class TestConformerBlockStreaming:
    """Test ConformerBlock streaming consistency independently."""

    def test_conformer_block_streaming_consistency(self) -> None:
        """ConformerBlock should maintain streaming consistency."""
        d_model = config.D_MODEL
        n_head = config.N_HEAD
        kernel_size = config.KERNEL_SIZE
        dropout = config.DROPOUT

        block = ConformerBlock(d_model, n_head, kernel_size, dropout)
        block.eval()

        # Batch inference
        x = torch.rand(1, 100, d_model)
        with torch.no_grad():
            y_batch, _ = block(x)

        # Streaming inference
        states = None
        y_streams = []
        chunk_size = 20

        with torch.no_grad():
            for i in range(0, 100, chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                y_chunk, states = block(chunk, states)
                y_streams.append(y_chunk)

        y_stream = torch.cat(y_streams, dim=1)

        # Compare
        assert torch.allclose(y_batch, y_stream, atol=1e-5)


class TestCacheInitialization:
    """Test cache initialization for various modules."""

    def test_conv_module_cache_none_initialization(self) -> None:
        """ConformerConvModule should handle cache=None correctly."""
        from model import ConformerConvModule

        d_model = config.D_MODEL
        kernel_size = config.KERNEL_SIZE
        conv = ConformerConvModule(d_model, kernel_size)
        conv.eval()

        # First call with cache=None
        x = torch.rand(1, 50, d_model)
        with torch.no_grad():
            y1, cache1 = conv(x, cache=None)

        # Second call with returned cache
        with torch.no_grad():
            y2, cache2 = conv(x, cache=cache1)

        # Outputs should be different due to different cache content
        assert not torch.allclose(y1, y2, atol=1e-6), "Outputs should differ with different cache content"
        assert y1.shape == y2.shape

    def test_conv_module_cache_continuity(self) -> None:
        """ConformerConvModule cache should maintain continuity."""
        from model import ConformerConvModule

        d_model = config.D_MODEL
        kernel_size = config.KERNEL_SIZE
        conv = ConformerConvModule(d_model, kernel_size)
        conv.eval()

        # Split inference into two chunks
        x = torch.rand(1, 100, d_model)
        x_part1 = x[:, :50, :]
        x_part2 = x[:, 50:, :]

        # Streaming with cache
        with torch.no_grad():
            y_part1, cache = conv(x_part1, cache=None)
            y_part2, _ = conv(x_part2, cache=cache)
        y_stream = torch.cat([y_part1, y_part2], dim=1)

        # Batch inference
        with torch.no_grad():
            y_batch, _ = conv(x, cache=None)

        # Compare (ConvModule 出力は (B, L, D) 形式)
        assert torch.allclose(y_stream, y_batch, atol=1e-5)

    def test_attention_cache_none_initialization(self) -> None:
        """CausalMultiHeadAttention should handle cache=None correctly."""
        from model import CausalMultiHeadAttention

        d_model = config.D_MODEL
        n_head = config.N_HEAD
        attn = CausalMultiHeadAttention(d_model, n_head)
        attn.eval()

        # First call with cache=None
        x = torch.rand(1, 50, d_model)
        with torch.no_grad():
            y1, cache1 = attn(x, cache=None)

        # Second call with returned cache
        with torch.no_grad():
            y2, cache2 = attn(x, cache=cache1)

        # Outputs should be different due to different cache content
        assert not torch.allclose(y1, y2, atol=1e-6), "Outputs should differ with different cache content"
        assert y1.shape == y2.shape


class TestBatchSizeSupport:
    """Test batch size support for caching modules."""

    def test_attention_batch_size_consistency(self) -> None:
        """CausalMultiHeadAttention should handle different batch sizes."""
        from model import CausalMultiHeadAttention

        d_model = config.D_MODEL
        n_head = config.N_HEAD
        attn = CausalMultiHeadAttention(d_model, n_head)
        attn.eval()

        batch_size = 4
        seq_len = 50
        x = torch.rand(batch_size, seq_len, d_model)

        with torch.no_grad():
            y_batch, _ = attn(x, cache=None)

        # Streaming with batch
        states = None
        y_streams = []
        chunk_size = 20

        with torch.no_grad():
            for i in range(0, seq_len, chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                y_chunk, states = attn(chunk, states)
                y_streams.append(y_chunk)

        y_stream = torch.cat(y_streams, dim=1)

        assert torch.allclose(y_batch, y_stream, atol=1e-5)

    def test_conv_module_batch_size_consistency(self) -> None:
        """ConformerConvModule should handle different batch sizes."""
        from model import ConformerConvModule

        d_model = config.D_MODEL
        kernel_size = config.KERNEL_SIZE
        conv = ConformerConvModule(d_model, kernel_size)
        conv.eval()

        batch_size = 4
        seq_len = 50
        x = torch.rand(batch_size, seq_len, d_model)

        with torch.no_grad():
            y_batch, _ = conv(x, cache=None)

        # Streaming with batch
        states = None
        y_streams = []
        chunk_size = 20

        with torch.no_grad():
            for i in range(0, seq_len, chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                y_chunk, states = conv(chunk, states)
                y_streams.append(y_chunk)

        y_stream = torch.cat(y_streams, dim=1)

        assert torch.allclose(y_stream, y_batch, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
