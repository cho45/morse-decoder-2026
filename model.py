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

    def extend_pe(self, length: int, device: torch.device = torch.device('cpu')):
        pe = torch.zeros(length, self.d_model, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
        self.max_len = length

    def forward(self, length: int, offset=0):
        """
        Args:
            length: シーケンス長 (Python int, x.size() から取得)
            offset: 開始位置 (テンソルまたは整数)
        """
        # offset をテンソルに変換
        if torch.is_tensor(offset):
            o = offset.reshape(())
        else:
            o = torch.tensor(offset, device=self.pe.device, dtype=torch.long)

        # トレーシング中はサイズ拡張を行わない
        if not torch.jit.is_tracing():
            o_val = int(o.item())
            if o_val + length > self.max_len:
                self.extend_pe(o_val + length + 1000, self.pe.device)

        # index_select を使用して ONNX 互換のスライスを行う
        indices = torch.arange(length, device=self.pe.device, dtype=torch.long) + o
        return self.pe.index_select(0, indices)

class CausalMultiHeadAttention(nn.Module):
    """Causal Multi-Head Attention."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        if cache is not None:
            prev_k, prev_v, offset = cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            new_offset = offset + seq_len
        else:
            new_offset = torch.as_tensor(seq_len, dtype=torch.long, device=x.device)
            
        # Limit cache size using unconditional slicing (ONNX compatible)
        # PyTorch negative slicing returns full tensor if size < MAX_CACHE_LEN
        k = k[:, :, -config.MAX_CACHE_LEN:, :]
        v = v[:, :, -config.MAX_CACHE_LEN:, :]
            
        new_cache = (k, v, new_offset)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask
        k_len = k.size(2)
        q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1) + (new_offset - seq_len)
        k_idx = torch.arange(k_len, device=x.device).unsqueeze(0)
        mask = q_idx < k_idx
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

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

        cache_len = self.kernel_size - 1
        if cache is not None:
            x = torch.cat([cache, x], dim=2)
        else:
            pad = torch.zeros(x.size(0), x.size(1), cache_len, device=x.device)
            x = torch.cat([pad, x], dim=2)
        
        # Save new cache
        new_cache = x[:, :, -cache_len:]

        x = self.depthwise_conv(x)
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
        self.ln_ff1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_head, dropout)
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size)
        self.ln_ff2 = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        attn_cache, conv_cache = cache if cache is not None else (None, None)
        x = x + 0.5 * self.ff1(self.ln_ff1(x))
        attn_out, new_attn_cache = self.attn(self.ln_attn(x), attn_cache)
        x = x + attn_out
        conv_out, new_conv_cache = self.conv(x, conv_cache)
        x = x + conv_out
        x = x + 0.5 * self.ff2(self.ln_ff2(x))
        x = self.ln_final(x)
        return x, (new_attn_cache, new_conv_cache)

class ConvSubsampling(nn.Module):
    """Strictly Causal Convolutional Subsampling (2x downsampling)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=3, stride=(2, 2), padding=(0, 1))
        f_out = (config.N_BINS + 2*1 - 3) // 2 + 1
        self.out_linear = nn.Linear(out_channels * f_out, out_channels)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1) # (B, 1, T, F)
        padding_t = 2
        if cache is None:
            # x.size(3) is the frequency dimension (N_BINS)
            pad = torch.zeros(x.size(0), 1, padding_t, x.size(3), device=x.device)
            x = torch.cat([pad, x], dim=2)
        else:
            x = torch.cat([cache, x], dim=2)
            
        l_in = x.size(2)
        n_out = (l_in - 3) // 2 + 1
        
        if not torch.jit.is_tracing() and n_out <= 0:
            return torch.zeros(x.size(0), 0, self.out_linear.out_features, device=x.device), x

        # n_out >= 1 が保証されているため l_consumed >= 3 は常に成立
        # (n_out=1 のとき l_consumed = 0*2+3 = 3)
        l_consumed = (n_out - 1) * 2 + 3
        x_valid = x[:, :, :l_consumed, :]
        
        cache_start = n_out * 2
        new_cache = x[:, :, cache_start:, :]
        
        x_out = F.relu(self.conv(x_valid))
        b, c, t, f = x_out.size()
        x_out = x_out.transpose(1, 2).contiguous().view(b, t, c * f)
        return self.out_linear(x_out), new_cache

class StreamingConformer(nn.Module):
    def __init__(self,
                 n_mels: int = config.N_BINS,
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
        self.final_dropout = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(d_model)
        self.signal_head = nn.Linear(d_model, config.NUM_SIGNAL_CLASSES)
        self.boundary_head = nn.Linear(d_model, 1)
        self.ctc_head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.ctc_head.weight, std=0.01)
        nn.init.constant_(self.ctc_head.bias, 0)
        with torch.no_grad():
            self.ctc_head.bias[0] = -5.0
        
        nn.init.trunc_normal_(self.signal_head.weight, std=0.01)
        nn.init.constant_(self.signal_head.bias, 0)

    def forward(self, x: torch.Tensor,
                states: Optional[Tuple[Optional[torch.Tensor], Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]]] = None) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]]:
        sub_cache, layer_states = states if states is not None else (None, None)
        x, new_sub_cache = self.subsampling(x, sub_cache)
        batch_size, seq_len, d_model = x.size()
        x = x * (d_model ** 0.5)
        
        if layer_states is not None:
            offset = layer_states[0][0][2]
        else:
            # offset を常にテンソルとして扱う（ONNX 互換）
            offset = torch.tensor(0, dtype=torch.long, device=x.device)
        
        pos_emb = self.pos_enc(seq_len, offset=offset)
        x = x + pos_emb
        
        new_layer_states = []
        for i, layer in enumerate(self.layers):
            cache = layer_states[i] if layer_states is not None else None
            x, new_cache = layer(x, cache)
            new_layer_states.append(new_cache)
            
        x = self.final_dropout(self.final_ln(x))
        signal_logits = self.signal_head(x)
        boundary_logits = self.boundary_head(x)
        logits = self.ctc_head(x)
        
        return (logits, signal_logits, boundary_logits), (new_sub_cache, new_layer_states)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = StreamingConformer(num_layers=4).to(device)
    model.eval()
    num_test_frames = config.SUBSAMPLING_RATE * 200
    dummy_input = torch.randn(1, num_test_frames, config.N_BINS).to(device)
    with torch.no_grad():
        (output, sig_out, bound_out), _ = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")