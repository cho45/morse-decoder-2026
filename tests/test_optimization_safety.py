import pytest
import torch
import config
from model import RelPositionalEncoding, PCEN, ConformerBlock


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
