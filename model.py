import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import config

class RelPositionalEncoding(nn.Module):
    """Relative Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = 5000): # Increased default max_len
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.extend_pe(max_len)

    def extend_pe(self, length: int):
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = length

    def forward(self, length: int, offset: int = 0):
        if offset + length > self.max_len:
            self.extend_pe(offset + length + 1000)
        return self.pe[offset : offset + length, :]

class CausalRelMultiHeadAttention(nn.Module):
    """Causal Multi-Head Attention with Relative Positional Encoding."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.pos_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, int]]:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        p = self.pos_linear(pos_emb).view(seq_len, self.n_head, self.d_k).transpose(0, 1)

        offset = 0
        if cache is not None:
            prev_k, prev_v, offset = cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            
            # Limit cache size to prevent infinite memory growth
            if k.size(2) > config.MAX_CACHE_LEN:
                k = k[:, :, -config.MAX_CACHE_LEN:, :]
                v = v[:, :, -config.MAX_CACHE_LEN:, :]
                offset = config.MAX_CACHE_LEN - seq_len
        
        new_offset = offset + seq_len
        new_cache = (k, v, new_offset)
        
        # Relative positional attention
        # q: (B, H, L, Dk), k: (B, H, L_total, Dk)
        # p is (H, L, Dk). We need (1, H, L, Dk) for broadcasting.
        
        # Standard Conformer RelAttn often uses (q + p) * k^T
        # Reshape p for broadcasting: (H, L, Dk) -> (1, H, L, Dk)
        p_unsqueezed = p.unsqueeze(0)
        q_with_pos = q + p_unsqueezed
        
        scores = torch.matmul(q_with_pos, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add causal mask if not streaming (or based on current seq_len)
        if seq_len > 1 or cache is None:
            mask = torch.triu(torch.ones(seq_len, k.size(2), device=x.device), diagonal=k.size(2)-seq_len+1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(x), new_cache

class ConformerConvModule(nn.Module):
    """Causal Convolution Module in Conformer."""
    def __init__(self, d_model: int, kernel_size: int = config.KERNEL_SIZE):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        # Causal padding: pad with kernel_size - 1 on the left
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                        padding=0, groups=d_model)
        # Use LayerNorm instead of GroupNorm/BatchNorm to avoid dependency on sequence length
        self.batch_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        x = self.layer_norm(x).transpose(1, 2) # (B, D, L)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1) # (B, D, L)

        # Causal Depthwise Conv with cache
        if cache is not None:
            x = torch.cat([cache, x], dim=2)
        
        # Save new cache (last kernel_size - 1 frames)
        cache_len = self.kernel_size - 1
        new_cache = x[:, :, -cache_len:] if x.size(2) >= cache_len else F.pad(x, (cache_len - x.size(2), 0))
        
        # Apply padding manually for the first chunk if no cache
        if cache is None:
            x = F.pad(x, (cache_len, 0))

        # Ensure x is long enough for depthwise_conv
        if x.size(2) < self.kernel_size:
            x = F.pad(x, (self.kernel_size - x.size(2), 0))

        x = self.depthwise_conv(x)
        # GroupNorm/BatchNorm expects (B, C, L), but LayerNorm expects (B, L, C)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2), new_cache

class ConformerBlock(nn.Module):
    """A single Conformer Block."""
    def __init__(self, d_model: int, n_head: int, kernel_size: int = config.KERNEL_SIZE, dropout: float = config.DROPOUT):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.attn = CausalRelMultiHeadAttention(d_model, n_head, dropout)
        self.conv = ConformerConvModule(d_model, kernel_size)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor, int], torch.Tensor]]:
        attn_cache, conv_cache = cache if cache is not None else (None, None)
        
        # FF1
        x = x + 0.5 * self.ff1(x)
        # MHSA
        x_ln = self.ln_attn(x)
        attn_out, new_attn_cache = self.attn(x_ln, pos_emb, attn_cache)
        x = x + attn_out
        # Conv
        conv_out, new_conv_cache = self.conv(x, conv_cache)
        x = x + conv_out
        # FF2
        x = x + 0.5 * self.ff2(x)
        x = self.ln_final(x)
        return x, (new_attn_cache, new_conv_cache)

class ConvSubsampling(nn.Module):
    """Strictly Causal Convolutional Subsampling (2x downsampling as per plan.md)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Downsample by 2x in both time and frequency
        # kernel_size=3, stride=2 requires 2 frames of padding for causality
        self.conv = nn.Conv2d(1, out_channels, kernel_size=3, stride=(2, 2))
        # F_in -> (F_in + 2*padding - kernel_size)//stride + 1
        # padding=1, kernel_size=3, stride=2
        f_out = (config.N_MELS + 2 - 3) // 2 + 1
        self.out_linear = nn.Linear(out_channels * f_out, out_channels)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        x = x.unsqueeze(1) # (B, 1, T, F)
        
        # Handle time-axis causality first (padding/caching)
        # We perform operations on x BEFORE frequency padding to keep cache clean
        
        if cache is not None:
            x = torch.cat([cache, x], dim=2)
        else:
            # Initial padding for causality (time axis)
            # kernel=3, stride=2. We need 2 frames padding at start.
            x = F.pad(x, (0, 0, 2, 0)) # freq_pad=0, time_pad=2
            
        l_in = x.size(2)
        kernel_size = 3
        stride = 2
        
        # Calculate valid output length
        n_out = (l_in - kernel_size) // stride + 1
        
        if n_out <= 0:
            new_cache = x
            # Return empty tensor with correct output dimension
            b, _, _, _ = x.size()
            return torch.zeros(b, 0, self.out_linear.out_features, device=x.device), new_cache

        # Determine new cache (unprocessed frames for next chunk)
        new_cache = x[:, :, n_out * stride :, :]
        
        # Determine valid input for convolution
        l_consumed = (n_out - 1) * stride + kernel_size
        x_valid = x[:, :, :l_consumed, :]
        
        # Apply frequency padding ONLY to the valid input for convolution
        x_valid = F.pad(x_valid, (1, 1, 0, 0)) # freq_pad=1 (sym)
        
        x_out = F.relu(self.conv(x_valid))
        
        b, c, t, f = x_out.size()
        x_out = x_out.transpose(1, 2).contiguous().view(b, t, c * f)
        return self.out_linear(x_out), new_cache

class StreamingConformer(nn.Module):
    def __init__(self,
                 n_mels: int = config.N_MELS,
                 num_classes: int = config.NUM_CLASSES,
                 d_model: int = config.D_MODEL,
                 n_head: int = config.N_HEAD,
                 num_layers: int = config.NUM_LAYERS,
                 kernel_size: int = config.KERNEL_SIZE,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d_model)
        self.pos_enc = RelPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_head, kernel_size, dropout) for _ in range(num_layers)
        ])
        self.ctc_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor,
                states: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor, int], torch.Tensor]]]]] = None) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], List[Tuple[Tuple[torch.Tensor, torch.Tensor, int], torch.Tensor]]]]:
        """
        x: (B, T, F)
        states: (subsampling_cache, layer_states)
        """
        sub_cache, layer_states = states if states is not None else (None, None)
        
        x, new_sub_cache = self.subsampling(x, sub_cache)
        batch_size, seq_len, d_model = x.size()
        
        # In streaming, we need to handle relative positional encoding carefully.
        # We need to know the global start position of the current chunk.
        # We can infer it from the attention cache of the first layer.
        offset = 0
        if layer_states is not None:
            # layer_states[0] is (attn_cache, conv_cache)
            # attn_cache is (k, v, offset)
            offset = layer_states[0][0][2]
        
        # Generate PE for current chunk with correct offset
        pos_emb = self.pos_enc(seq_len, offset=offset)
        
        new_layer_states = []
        for i, layer in enumerate(self.layers):
            cache = layer_states[i] if layer_states is not None else None
            x, new_cache = layer(x, pos_emb, cache)
            new_layer_states.append(new_cache)
            
        logits = self.ctc_head(x)
        return logits, (new_sub_cache, new_layer_states)

if __name__ == "__main__":
    # Test the model
    device = torch.device("cpu")
    model = StreamingConformer(
        n_mels=config.N_MELS,
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        num_layers=4
    ).to(device)
    model.eval()
    
    print("Testing batch inference...")
    num_test_frames = config.SUBSAMPLING_RATE * 200
    dummy_input = torch.randn(1, num_test_frames, config.N_MELS).to(device)
    with torch.no_grad():
        output, _ = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting streaming inference...")
    # Chunk size must be multiple of 4 for this subsampling implementation
    chunk_size = 40
    states = None
    all_outputs = []
    
    with torch.no_grad():
        for i in range(0, num_test_frames, chunk_size):
            chunk = dummy_input[:, i:i+chunk_size, :]
            logits, states = model(chunk, states)
            all_outputs.append(logits)
            print(f"Chunk {i//chunk_size} processed. Output shape: {logits.shape}")
            
    full_streaming_output = torch.cat(all_outputs, dim=1)
    print(f"Full streaming output shape: {full_streaming_output.shape}")
    
    # Check if shapes match
    assert full_streaming_output.shape == output.shape, f"Shape mismatch: {full_streaming_output.shape} vs {output.shape}"
    
    # Check if values match (within tolerance)
    diff = torch.abs(full_streaming_output - output).max().item()
    print(f"Max difference between batch and streaming: {diff:.6f}")
    
    if diff < 1e-5:
        print("\nStreaming test passed! Output values match.")
    else:
        print("\nStreaming test WARNING: Values do not match perfectly. This might be due to BatchNormalization or Subsampling padding.")