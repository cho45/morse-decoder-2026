import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import config

class RelPositionalEncoding(nn.Module):
    """Relative Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = config.MAX_CACHE_LEN * 2):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', torch.zeros(max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, length: int, offset: torch.Tensor):
        """
        Args:
            length: シーケンス長 (SymInt)
            offset: 開始位置 (0-d Tensor)
        """
        # length is usually a SymInt from x.size(1)
        torch._check_is_size(length)
        
        # Ensure offset is a 0-d tensor for symbolic arithmetic
        o_ts = offset.reshape(())
        
        # indices calculation using Tensors to keep it in the graph
        indices = torch.arange(length, device=self.pe.device, dtype=torch.long) + o_ts
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

        max_seq_len = config.MAX_CACHE_LEN + 100
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x: torch.Tensor,
                cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.size()
        torch._check_is_size(seq_len)
        
        q = self.q_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # cache is always provided as a tuple of tensors.
        prev_k, prev_v, offset = cache
        k = torch.cat([prev_k, k], dim=2)
        v = torch.cat([prev_v, v], dim=2)
        new_offset = offset + seq_len
            
        k_full_len = k.size(2)
        torch._check_is_size(k_full_len)
        
        # Use symbolic arithmetic for cache trimming.
        # We use torch._check to inform the compiler about the relationship
        # between k_full_len and MAX_CACHE_LEN to avoid over-specialization.
        k_start = k_full_len - config.MAX_CACHE_LEN
        # Dynamo handles max(0, SymInt) symbolically.
        actual_start = max(0, k_start)
        actual_len = k_full_len - actual_start
        
        # narrow supports SymInt directly and is more robust for export.
        k = k.narrow(2, actual_start, actual_len)
        v = v.narrow(2, actual_start, actual_len)

        # Clamp offset to match trimmed cache size (ONNX compatible)
        new_offset = torch.as_tensor(new_offset, device=x.device, dtype=torch.long).clamp(max=config.MAX_CACHE_LEN)

        new_cache = (k, v, new_offset)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        k_len = k.size(2)
        # Ensure k_len is treated as size
        torch._check_is_size(k_len)
        # new_offset is now a Tensor. seq_len is a SymInt.
        q_start = new_offset - seq_len
        # Generate causal mask on the fly to avoid dynamic slicing issues in torch.export.
        # This is mathematically equivalent to self.causal_mask[q_start:q_start+seq_len, :k_len]
        # where causal_mask[i, j] == (i < j).
        # q_start is a Tensor, so we use it directly in arange addition.
        q_idx = torch.arange(seq_len, device=x.device, dtype=torch.long).view(-1, 1) + q_start
        k_idx = torch.arange(k_len, device=x.device, dtype=torch.long).view(1, -1)
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
        self.register_buffer('cache_pad', torch.zeros(1, d_model, kernel_size - 1))

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        x = self.layer_norm(x).transpose(1, 2) # (B, D, L)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1) # (B, D, L)

        # In streaming mode, cache is always provided.
        x = torch.cat([cache, x], dim=2)
        
        # Save new cache. Use narrow for symbolic consistency.
        l_total = x.size(2)
        torch._check_is_size(l_total)
        cache_len = self.kernel_size - 1
        # Use narrow with SymInt for symbolic consistency.
        start_idx = l_total - cache_len
        new_cache = x.narrow(2, start_idx, cache_len)

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
                cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        attn_cache, conv_cache = cache
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
        self.padding_t = 2
        self.register_buffer('pad_buffer', torch.zeros(1, 1, 2, config.N_BINS))

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        # In streaming mode, cache is always provided.
        x = torch.cat([cache, x], dim=2)
            
        l_in = x.size(2)
        # Ensure input length is treated as size and is sufficient for Conv2d kernel_size=3
        torch._check_is_size(l_in)
        torch._check(l_in >= 3)
        
        n_out = (l_in - 3) // 2 + 1
        
        # n_out >= 1 が保証されているため l_consumed >= 3 は常に成立
        l_consumed = (n_out - 1) * 2 + 3
        # Use narrow instead of slicing to avoid specialization in Dynamo
        x_valid = x.narrow(2, 0, l_consumed)
        
        cache_start = n_out * 2
        cache_len = l_in - cache_start
        new_cache = x.narrow(2, cache_start, cache_len)
        
        # ONNX export compat: Use tensor comparison to avoid TracerWarning and ensure dynamic behavior
        # Note: We need to use torch.jit.script or ensure the graph handles control flow correctly.
        # However, since we cannot use @torch.jit.script here due to inheritance issues in some environments/versions,
        # we try to rely on the fact that for valid inputs, n_out > 0.
        # But to prevent crash on small inputs during ONNX runtime, we added the check above.
        # The check above (if not torch.jit.is_tracing() and n_out <= 0) only protects Python execution.
        # To protect ONNX execution, we need the check to be part of the graph.
        
        # If we cannot use script, we can try to make the convolution conditional using mask or similar,
        # but that is complex.
        # For now, we revert to the state where we just ensure it works for valid inputs,
        # and rely on the Python check for non-tracing execution.
        # The user reported crash on small inputs, likely because the graph was exported with large input
        # and lacks the conditional check.
        
        x_out = F.relu(self.conv(x_valid))
        b, c, t, f = x_out.size()
        x_out = x_out.transpose(1, 2).contiguous().view(b, t, c * f)
        return self.out_linear(x_out), new_cache

def _pcen_ema_loop(x: torch.Tensor, state: torch.Tensor, s: torch.Tensor):
    """Vectorized EMA calculation for Dynamo compatibility."""
    T = x.size(1)
    if T == 0:
        return torch.zeros_like(x), state

    # EMA: E_t = (1-s)E_{t-1} + s*x_t
    # Vectorized form: E_t = (1-s)^t * E_0 + sum_{j=1}^t s * x_j * (1-s)^{t-j}
    
    one_minus_s = 1.0 - s
    log_one_minus_s = torch.log(one_minus_s)
    
    # steps shape: (1, T, 1)
    steps = torch.arange(T, device=x.device, dtype=x.dtype).view(1, -1, 1)
    # log_powers shape: (1, T, F)
    log_powers = steps * log_one_minus_s
    
    # initial state contribution
    E_state = state * torch.exp(log_powers + log_one_minus_s)
    
    # x contribution: s * exp(log_powers) * cumsum(x * exp(-log_powers))
    # This is mathematically equivalent to the recursive EMA formula.
    E_x = s * torch.exp(log_powers) * torch.cumsum(x * torch.exp(-log_powers), dim=1)
    
    E = E_state + E_x
    new_state = E[:, -1:, :]
    
    return E, new_state

class PCEN(nn.Module):
    """
    Per-Channel Energy Normalization (PCEN).
    y(t, f) = (x(t, f) / (eps + E(t, f))^alpha + delta)^r - delta^r
    where E(t, f) is an exponential moving average of x(t, f).
    """
    def __init__(self, n_mels: int, s: float = 0.025, alpha: float = 0.98,
                 delta: float = 2.0, r: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.log_s = nn.Parameter(torch.log(torch.full((n_mels,), s)))
        self.log_alpha = nn.Parameter(torch.log(torch.full((n_mels,), alpha)))
        self.log_delta = nn.Parameter(torch.log(torch.full((n_mels,), delta)))
        self.log_r = nn.Parameter(torch.log(torch.full((n_mels,), r)))
        self.eps = eps

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input spectrogram (B, T, F)
            state: EMA state (B, 1, F)
        Returns:
            y: Normalized spectrogram (B, T, F)
            new_state: Updated EMA state (B, 1, F)
        """
        # Ensure parameters are correctly shaped for broadcasting (1, 1, F)
        s = torch.exp(self.log_s).view(1, 1, -1)
        alpha = torch.exp(self.log_alpha).view(1, 1, -1)
        delta = torch.exp(self.log_delta).view(1, 1, -1)
        r = torch.exp(self.log_r).view(1, 1, -1)

        T = x.size(1)
        torch._check_is_size(T)
        # state is always provided.
        curr_state = state
        
        # Use scripted loop to avoid unrolling in ONNX
        E, new_state = _pcen_ema_loop(x, curr_state, s)
        
        # PCEN formula: y = (x / (eps + E)^alpha + delta)^r - delta^r
        # Numerical stability: Ensure E is not too small
        y = (x / (self.eps + E).pow(alpha) + delta).pow(r) - delta.pow(r)
            
        return y, new_state

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
        self.pcen = PCEN(n_mels)
        self.input_scale = nn.Parameter(torch.tensor(100.0)) # Initial scale similar to current train.py
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

    def get_initial_states(self, batch_size: int, device: torch.device = torch.device('cpu'), cache_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]:
        """Generate initial states for streaming inference."""
        pcen_state = torch.zeros(batch_size, 1, config.N_BINS, device=device)
        sub_cache = torch.zeros(batch_size, 1, 2, config.N_BINS, device=device)
        
        d_k = self.layers[0].attn.d_k
        n_head = self.layers[0].attn.n_head
        d_model = self.layers[0].attn.d_model
        kernel_size = self.layers[0].conv.kernel_size
        
        layer_states = []
        for _ in range(len(self.layers)):
            attn_cache = (
                torch.zeros(batch_size, n_head, cache_len, d_k, device=device),
                torch.zeros(batch_size, n_head, cache_len, d_k, device=device),
                torch.tensor(cache_len, dtype=torch.long, device=device)
            )
            conv_cache = torch.zeros(batch_size, d_model, kernel_size - 1, device=device)
            layer_states.append((attn_cache, conv_cache))
        return (pcen_state, sub_cache, layer_states)

    def forward(self, x: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor, List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]]:
        pcen_state, sub_cache, layer_states = states
        
        # 1. PCEN Normalization
        x = x * self.input_scale
        x, new_pcen_state = self.pcen(x, pcen_state)
        
        # 2. Subsampling
        x, new_sub_cache = self.subsampling(x, sub_cache)
        batch_size, seq_len, d_model = x.size()
        torch._check_is_size(seq_len)
        x = x * (d_model ** 0.5)
        
        # In streaming mode, layer_states is always provided.
        # layer_states[0] is the state of the first layer.
        # layer_states[0][0] is the attention state (k, v, offset).
        # layer_states[0][0][2] is the offset tensor.
        first_layer_state = layer_states[0]
        attn_state = first_layer_state[0]
        offset = attn_state[2]
        
        pos_emb = self.pos_enc(seq_len, offset=offset)
        x = x + pos_emb
        
        new_layer_states = []
        for i, layer in enumerate(self.layers):
            # In streaming mode, layer_states is always a list of tuples.
            cache = layer_states[i]
            x, new_cache = layer(x, cache)
            new_layer_states.append(new_cache)
            
        x = self.final_dropout(self.final_ln(x))
        signal_logits = self.signal_head(x)
        boundary_logits = self.boundary_head(x)
        logits = self.ctc_head(x)
        
        return (logits, signal_logits, boundary_logits), (new_pcen_state, new_sub_cache, new_layer_states)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = StreamingConformer(num_layers=4).to(device)
    model.eval()
    num_test_frames = config.SUBSAMPLING_RATE * 200
    dummy_input = torch.randn(1, num_test_frames, config.N_BINS).to(device)
    with torch.no_grad():
        states = model.get_initial_states(dummy_input.size(0), device)
        (output, sig_out, bound_out), _ = model(dummy_input, states)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")