"""
ONNX Export Tests for CW Decoder Model.

TDD approach: These tests define the expected behavior for ONNX export.
All tests should pass after model.py fixes are applied.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import tempfile
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import (
    RelPositionalEncoding,
    CausalMultiHeadAttention,
    ConformerConvModule,
    ConvSubsampling,
    StreamingConformer,
)
from export_onnx import ONNXWrapper


def create_initial_states(batch_size, num_layers, device='cpu', example_cache_len=0):
    """Create initial cache states."""
    d_k = config.D_MODEL // config.N_HEAD
    pcen_state = torch.zeros(batch_size, 1, config.N_BINS, device=device)
    sub_cache = torch.zeros(batch_size, 1, 2, config.N_BINS, device=device)

    layer_states_flat = []
    for _ in range(num_layers):
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, example_cache_len, d_k, device=device))  # k
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, example_cache_len, d_k, device=device))  # v
        # Use scalar tensor for offset to prevent specialization in torch.export
        layer_states_flat.append(torch.tensor(example_cache_len, dtype=torch.long, device=device))  # offset
        layer_states_flat.append(torch.zeros(batch_size, config.D_MODEL, config.KERNEL_SIZE - 1, device=device))  # conv

    return pcen_state, sub_cache, layer_states_flat


def export_model_to_onnx(model, onnx_path, seq_len=12):
    """Export model to ONNX format."""
    wrapper = ONNXWrapper(model)
    wrapper.eval()  # Ensure wrapper is in eval mode
    batch_size = 2
    # Use positive inputs for PCEN
    x = torch.rand(batch_size, seq_len, config.N_BINS)
    # Use non-zero, non-one cache length for example inputs to avoid 0/1 specialization issues.
    pcen_state, sub_cache, layer_states_flat = create_initial_states(batch_size, len(model.layers), example_cache_len=10)

    input_names = ['x', 'pcen_state', 'sub_cache']
    output_names = ['logits', 'signal_logits', 'boundary_logits', 'new_pcen_state', 'new_sub_cache']

    for i in range(len(model.layers)):
        input_names.extend([f'attn_k_{i}', f'attn_v_{i}', f'offset_{i}', f'conv_cache_{i}'])
        output_names.extend([f'new_attn_k_{i}', f'new_attn_v_{i}', f'new_offset_{i}', f'new_conv_cache_{i}'])

    # Define dynamic shapes for torch.export
    batch = torch.export.Dim("batch", min=1, max=4)
    # seq=1, sub_cache_len=0 だと n_out=0 になり ConvSubsampling の if に引っかかるため
    # 最小値を調整して n_out > 0 を保証する。
    # また ONNXWrapper の torch._check (Batch * n_out < 80) と整合させる。
    seq = torch.export.Dim("seq", min=3, max=40)
    sub_cache_len = torch.export.Dim("sub_cache_len", min=0, max=100)
    # Use config.MAX_CACHE_LEN for the dynamic range.
    attn_cache_len = torch.export.Dim("attn_cache_len", min=0, max=config.MAX_CACHE_LEN)
    
    dynamic_shapes = {
        "x": {0: batch, 1: seq},
        "pcen_state": {0: batch},
        "sub_cache": {0: batch, 2: sub_cache_len},
    }
    for i in range(config.NUM_LAYERS):
        dynamic_shapes[f"attn_k_{i}"] = {0: batch, 2: attn_cache_len}
        dynamic_shapes[f"attn_v_{i}"] = {0: batch, 2: attn_cache_len}
        # For scalar tensors, an empty dict {} marks them as dynamic in torch.export.
        dynamic_shapes[f"offset_{i}"] = {}
        dynamic_shapes[f"conv_cache_{i}"] = {0: batch}

    # Step 1: Create an ExportedProgram with explicit constraints
    # Use strict=False to avoid over-specialization on example inputs
    exported_program = torch.export.export(
        wrapper,
        args=(x, pcen_state, sub_cache, *layer_states_flat),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

    # Step 2: Convert ExportedProgram to ONNX
    torch.onnx.export(
        exported_program,
        args=(), # ExportedProgram already contains example inputs
        f=onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=True
    )

    return wrapper, input_names, output_names


def to_numpy(t):
    """Convert tensor or scalar to numpy array."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return np.array(t)
    return t.detach().cpu().numpy()


class TestONNXExportNoWarnings:
    """Test that ONNX export produces no TracerWarnings."""

    def test_onnx_export_no_tracer_warnings(self):
        """ONNX export should produce no TracerWarnings."""
        model = StreamingConformer(num_layers=4)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Use positive inputs for PCEN during export
                export_model_to_onnx(model, onnx_path)

                tracer_warnings = [
                    warning for warning in w
                    if 'TracerWarning' in str(warning.category.__name__)
                ]

                if tracer_warnings:
                    warning_messages = [str(warning.message) for warning in tracer_warnings]
                    pytest.fail(f"TracerWarnings detected:\n" + "\n".join(warning_messages))
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


class TestONNXOutputEquivalence:
    """Test that ONNX model outputs match PyTorch outputs."""

    @pytest.fixture(scope="class")
    def model_and_session(self):
        """Create model and ONNX session once per test class."""
        model = StreamingConformer(num_layers=4)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            yield model, wrapper, session, input_names, output_names
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    @pytest.mark.parametrize("seq_len", [10, 20, 30, 40])
    def test_onnx_equivalence_single_batch(self, model_and_session, seq_len):
        """ONNX model output must match PyTorch within 1e-4."""
        model, wrapper, session, input_names, output_names = model_and_session
    
        batch_size = 1
        # Use positive inputs for PCEN
        x = torch.rand(batch_size, seq_len, config.N_BINS)
        pcen_state, sub_cache, layer_states_flat = create_initial_states(batch_size, len(model.layers))

        # PyTorch inference
        with torch.no_grad():
            pt_outputs = wrapper(x, pcen_state, sub_cache, *layer_states_flat)

        # ONNX inference
        ort_inputs = {
            'x': to_numpy(x),
            'pcen_state': to_numpy(pcen_state),
            'sub_cache': to_numpy(sub_cache)
        }
        for i in range(len(model.layers)):
            ort_inputs[f'attn_k_{i}'] = to_numpy(layer_states_flat[i*4])
            ort_inputs[f'attn_v_{i}'] = to_numpy(layer_states_flat[i*4+1])
            ort_inputs[f'offset_{i}'] = to_numpy(layer_states_flat[i*4+2])
            ort_inputs[f'conv_cache_{i}'] = to_numpy(layer_states_flat[i*4+3])

        ort_outputs = session.run(None, ort_inputs)

        # Compare outputs
        for i, name in enumerate(output_names):
            pt_out = to_numpy(pt_outputs[i])
            ort_out = ort_outputs[i]
            diff = np.abs(pt_out - ort_out).max()
            assert diff < 1e-4, f"Output '{name}' max diff: {diff:.6e} (expected < 1e-4)"


class TestONNXComponentEquivalence:
    """Individual component equivalence tests to pinpoint numerical issues."""
    
    def test_pcen_equivalence(self):
        from model import PCEN
        model = PCEN(config.N_BINS).eval()
        
        class PCENWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x, state):
                # state is always provided
                return self.m(x, state)
        
        wrapper = PCENWrapper(model).eval()
        x = torch.rand(1, 20, config.N_BINS)
        state = torch.rand(1, 1, config.N_BINS)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            export_output = torch.onnx.export(
                wrapper, (x, state), f.name,
                input_names=['x', 'state'],
                dynamic_shapes={'x': {1: 'T'}, 'state': {}},
                dynamo=True
            )
            session = ort.InferenceSession(f.name, providers=['CPUExecutionProvider'])
            
        pt_out, pt_state = wrapper(x, state)
        ort_outs = session.run(None, {'x': to_numpy(x), 'state': to_numpy(state)})
        
        diff_out = np.abs(to_numpy(pt_out) - ort_outs[0]).max()
        diff_state = np.abs(to_numpy(pt_state) - ort_outs[1]).max()
        
        assert diff_out < 1e-4, f"PCEN output diff too large: {diff_out:.6e}"
        assert diff_state < 1e-4, f"PCEN state diff too large: {diff_state:.6e}"

    def test_subsampling_equivalence(self):
        from model import ConvSubsampling
        model = ConvSubsampling(config.N_BINS, config.D_MODEL).eval()
        
        class SubWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x, cache):
                # In streaming mode, cache is always provided.
                l_in = x.size(1) + cache.size(2)
                torch._check_is_size(l_in)
                return self.m(x, cache)
                
        wrapper = SubWrapper(model).eval()
        x = torch.rand(1, 10, config.N_BINS)
        cache = torch.zeros(1, 1, 2, config.N_BINS)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(
                wrapper, (x, cache), f.name,
                input_names=['x', 'cache'],
                dynamic_shapes={'x': {1: 'T'}, 'cache': {}},
                dynamo=True
            )
            session = ort.InferenceSession(f.name, providers=['CPUExecutionProvider'])
            
        pt_out, pt_cache = wrapper(x, cache)
        ort_outs = session.run(None, {'x': to_numpy(x), 'cache': to_numpy(cache)})
        
        diff_out = np.abs(to_numpy(pt_out) - ort_outs[0]).max()
        assert diff_out < 1e-4, f"Subsampling output diff too large: {diff_out:.6e}"

    def test_attention_equivalence(self):
        from model import CausalMultiHeadAttention
        model = CausalMultiHeadAttention(config.D_MODEL, config.N_HEAD).eval()
        
        class AttnWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x, k, v, offset):
                # offset is a Tensor
                return self.m(x, (k, v, offset))
                
        wrapper = AttnWrapper(model).eval()
        batch_size = 1
        seq_len = 10
        cache_len = 20
        x = torch.rand(batch_size, seq_len, config.D_MODEL)
        k = torch.rand(batch_size, config.N_HEAD, cache_len, config.D_MODEL // config.N_HEAD)
        v = torch.rand(batch_size, config.N_HEAD, cache_len, config.D_MODEL // config.N_HEAD)
        offset = torch.tensor(cache_len, dtype=torch.long)
        
        # Define dynamic shapes including offset
        T = torch.export.Dim("T", min=1, max=100)
        C = torch.export.Dim("C", min=0, max=config.MAX_CACHE_LEN)
        O = torch.export.Dim("O", min=0, max=config.MAX_CACHE_LEN)
    
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(
                wrapper, (x, k, v, offset), f.name,
                input_names=['x', 'k', 'v', 'offset'],
                # offset is a scalar, so we use an empty dict or Dim.AUTO
                dynamic_shapes={'x': {1: T}, 'k': {2: C}, 'v': {2: C}, 'offset': {}},
                dynamo=True
            )
            session = ort.InferenceSession(f.name, providers=['CPUExecutionProvider'])
            
        pt_out, (pt_k, pt_v, pt_off) = wrapper(x, k, v, offset)
        ort_outs = session.run(None, {
            'x': to_numpy(x), 'k': to_numpy(k), 'v': to_numpy(v), 'offset': to_numpy(offset)
        })
        
        diff_out = np.abs(to_numpy(pt_out) - ort_outs[0]).max()
        assert diff_out < 1e-4, f"Attention output diff too large: {diff_out:.6e}"

class TestONNXStreamingEquivalence:
    """Test streaming inference equivalence between PyTorch and ONNX."""

    @pytest.fixture(scope="class")
    def model_and_session(self):
        """Create model and ONNX session once per test class."""
        model = StreamingConformer(num_layers=4)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            yield model, wrapper, session, len(model.layers), output_names
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    @pytest.mark.parametrize("chunk_size", [4, 8, 12, 16])
    def test_onnx_streaming_equivalence(self, model_and_session, chunk_size):
        """ONNX streaming with cache must match PyTorch streaming."""
        model, wrapper, session, num_layers, output_names = model_and_session

        batch_size = 1
        total_seq_len = 200
        # Use positive inputs for PCEN
        x_full = torch.rand(batch_size, total_seq_len, config.N_BINS)

        # Initialize states
        pt_pcen_state, pt_sub_cache, pt_layer_states_flat = create_initial_states(batch_size, num_layers)
        ort_pcen_state, ort_sub_cache, ort_layer_states_flat = create_initial_states(batch_size, num_layers)

        # Process in chunks
        for start in range(0, total_seq_len, chunk_size):
            end = min(start + chunk_size, total_seq_len)
            x_chunk = x_full[:, start:end, :]

            # PyTorch
            with torch.no_grad():
                pt_outputs = wrapper(x_chunk, pt_pcen_state, pt_sub_cache, *pt_layer_states_flat)

            # ONNX
            ort_inputs = {
                'x': to_numpy(x_chunk),
                'pcen_state': to_numpy(ort_pcen_state),
                'sub_cache': to_numpy(ort_sub_cache)
            }
            for i in range(num_layers):
                ort_inputs[f'attn_k_{i}'] = to_numpy(ort_layer_states_flat[i*4])
                ort_inputs[f'attn_v_{i}'] = to_numpy(ort_layer_states_flat[i*4+1])
                ort_inputs[f'offset_{i}'] = to_numpy(ort_layer_states_flat[i*4+2])
                ort_inputs[f'conv_cache_{i}'] = to_numpy(ort_layer_states_flat[i*4+3])

            ort_outputs = session.run(None, ort_inputs)

            # Compare outputs for this chunk
            for i in range(3):  # logits, signal_logits, boundary_logits
                pt_out = to_numpy(pt_outputs[i])
                ort_out = ort_outputs[i]
                diff = np.abs(pt_out - ort_out).max()
                assert diff < 1e-4, f"Chunk {start}-{end}, output {output_names[i]} max diff: {diff:.6e}"

            # Update states for next iteration
            pt_pcen_state = pt_outputs[3]
            pt_sub_cache = pt_outputs[4]
            pt_layer_states_flat = list(pt_outputs[5:])

            ort_pcen_state = torch.from_numpy(ort_outputs[3])
            ort_sub_cache = torch.from_numpy(ort_outputs[4])
            ort_layer_states_flat = [torch.from_numpy(o) for o in ort_outputs[5:]]


class TestONNXCacheStatesEquivalence:
    """Test that cache states from ONNX match PyTorch."""

    def test_onnx_cache_states_equivalence(self):
        """Cache states from ONNX must match PyTorch cache states."""
        model = StreamingConformer(num_layers=4)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            batch_size = 1
            chunk_size = 12
            num_chunks = 5

            pt_pcen_state, pt_sub_cache, pt_layer_states_flat = create_initial_states(batch_size, len(model.layers))
            ort_pcen_state, ort_sub_cache, ort_layer_states_flat = create_initial_states(batch_size, len(model.layers))

            for chunk_idx in range(num_chunks):
                # Use positive inputs for PCEN
                x_chunk = torch.rand(batch_size, chunk_size, config.N_BINS)

                # PyTorch
                with torch.no_grad():
                    pt_outputs = wrapper(x_chunk, pt_pcen_state, pt_sub_cache, *pt_layer_states_flat)

                # ONNX
                ort_inputs = {
                    'x': to_numpy(x_chunk),
                    'pcen_state': to_numpy(ort_pcen_state),
                    'sub_cache': to_numpy(ort_sub_cache)
                }
                for i in range(len(model.layers)):
                    ort_inputs[f'attn_k_{i}'] = to_numpy(ort_layer_states_flat[i*4])
                    ort_inputs[f'attn_v_{i}'] = to_numpy(ort_layer_states_flat[i*4+1])
                    ort_inputs[f'offset_{i}'] = to_numpy(ort_layer_states_flat[i*4+2])
                    ort_inputs[f'conv_cache_{i}'] = to_numpy(ort_layer_states_flat[i*4+3])

                ort_outputs = session.run(None, ort_inputs)

                # Compare cache states
                # new_pcen_state
                pt_pcen_np = to_numpy(pt_outputs[3])
                ort_pcen_np = ort_outputs[3]
                diff_pcen = np.abs(pt_pcen_np - ort_pcen_np).max()
                assert diff_pcen < 1e-4, f"Chunk {chunk_idx}, pcen_state max diff: {diff_pcen:.6e}"

                # new_sub_cache
                pt_sub_np = to_numpy(pt_outputs[4])
                ort_sub_np = ort_outputs[4]
                diff = np.abs(pt_sub_np - ort_sub_np).max()
                assert diff < 1e-4, f"Chunk {chunk_idx}, sub_cache max diff: {diff:.6e}"

                # Layer states
                for layer_idx in range(len(model.layers)):
                    base_pt = 5 + layer_idx * 4
                    base_ort = 5 + layer_idx * 4

                    # attn_k
                    diff_k = np.abs(to_numpy(pt_outputs[base_pt]) - ort_outputs[base_ort]).max()
                    assert diff_k < 1e-4, f"Chunk {chunk_idx}, layer {layer_idx} attn_k max diff: {diff_k:.6e}"

                    # attn_v
                    diff_v = np.abs(to_numpy(pt_outputs[base_pt+1]) - ort_outputs[base_ort+1]).max()
                    assert diff_v < 1e-4, f"Chunk {chunk_idx}, layer {layer_idx} attn_v max diff: {diff_v:.6e}"

                    # offset
                    diff_offset = np.abs(to_numpy(pt_outputs[base_pt+2]) - ort_outputs[base_ort+2]).max()
                    assert diff_offset < 1e-4, f"Chunk {chunk_idx}, layer {layer_idx} offset max diff: {diff_offset:.6e}"

                    # conv_cache
                    diff_conv = np.abs(to_numpy(pt_outputs[base_pt+3]) - ort_outputs[base_ort+3]).max()
                    assert diff_conv < 1e-4, f"Chunk {chunk_idx}, layer {layer_idx} conv_cache max diff: {diff_conv:.6e}"

                # Update states
                pt_pcen_state = pt_outputs[3]
                pt_sub_cache = pt_outputs[4]
                pt_layer_states_flat = list(pt_outputs[5:])

                ort_pcen_state = torch.from_numpy(ort_outputs[3])
                ort_sub_cache = torch.from_numpy(ort_outputs[4])
                ort_layer_states_flat = [torch.from_numpy(o) for o in ort_outputs[5:]]

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


class TestONNXCacheLimitBehavior:
    """Test ONNX model behavior when cache exceeds MAX_CACHE_LEN."""

    def test_onnx_cache_limit_behavior(self):
        """ONNX model must handle cache overflow correctly."""
        model = StreamingConformer(num_layers=4)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            batch_size = 1
            chunk_size = 16
            # Process enough chunks to exceed MAX_CACHE_LEN
            num_chunks = (config.MAX_CACHE_LEN // chunk_size) + 5

            pt_pcen_state, pt_sub_cache, pt_layer_states_flat = create_initial_states(batch_size, len(model.layers))
            ort_pcen_state, ort_sub_cache, ort_layer_states_flat = create_initial_states(batch_size, len(model.layers))

            for chunk_idx in range(num_chunks):
                # Use positive inputs for PCEN
                x_chunk = torch.rand(batch_size, chunk_size, config.N_BINS)

                # PyTorch
                with torch.no_grad():
                    pt_outputs = wrapper(x_chunk, pt_pcen_state, pt_sub_cache, *pt_layer_states_flat)

                # ONNX
                ort_inputs = {
                    'x': to_numpy(x_chunk),
                    'pcen_state': to_numpy(ort_pcen_state),
                    'sub_cache': to_numpy(ort_sub_cache)
                }
                for i in range(len(model.layers)):
                    ort_inputs[f'attn_k_{i}'] = to_numpy(ort_layer_states_flat[i*4])
                    ort_inputs[f'attn_v_{i}'] = to_numpy(ort_layer_states_flat[i*4+1])
                    ort_inputs[f'offset_{i}'] = to_numpy(ort_layer_states_flat[i*4+2])
                    ort_inputs[f'conv_cache_{i}'] = to_numpy(ort_layer_states_flat[i*4+3])

                ort_outputs = session.run(None, ort_inputs)

                # Verify cache size doesn't exceed MAX_CACHE_LEN
                for layer_idx in range(len(model.layers)):
                    base_ort = 5 + layer_idx * 4
                    k_cache = ort_outputs[base_ort]
                    assert k_cache.shape[2] <= config.MAX_CACHE_LEN, \
                        f"Chunk {chunk_idx}, layer {layer_idx} cache size {k_cache.shape[2]} exceeds {config.MAX_CACHE_LEN}"

                # Compare outputs
                for i in range(3):
                    pt_out = to_numpy(pt_outputs[i])
                    ort_out = ort_outputs[i]
                    diff = np.abs(pt_out - ort_out).max()
                    assert diff < 1e-4, f"Chunk {chunk_idx}, output {output_names[i]} max diff: {diff:.6e}"

                # Update states
                pt_pcen_state = pt_outputs[3]
                pt_sub_cache = pt_outputs[4]
                pt_layer_states_flat = list(pt_outputs[5:])

                ort_pcen_state = torch.from_numpy(ort_outputs[3])
                ort_sub_cache = torch.from_numpy(ort_outputs[4])
                ort_layer_states_flat = [torch.from_numpy(o) for o in ort_outputs[5:]]

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


class TestRelPositionalEncodingONNX:
    """Test RelPositionalEncoding ONNX compatibility."""

    def test_positional_encoding_onnx_equivalence(self):
        """RelPositionalEncoding must trace correctly with tensor offset.

        In real model usage:
        - length comes from x.size(1) which is a Python int
        - offset is a tensor from cache or newly created
        """

        class PEWrapper(nn.Module):
            """Wrapper that mimics real model PE usage."""
            def __init__(self, pe):
                super().__init__()
                self.pe = pe

            def forward(self, x, offset):
                # length is x.size(1) - a Python int during tracing
                length = x.size(1)
                return self.pe(length, offset)

        pe = RelPositionalEncoding(config.D_MODEL, max_len=5000)
        wrapper = PEWrapper(pe)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            x_export = torch.rand(1, 50, config.D_MODEL)
            offset = torch.tensor(0, dtype=torch.long)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.onnx.export(
                    wrapper,
                    (x_export, offset),
                    onnx_path,
                    input_names=['x', 'offset'],
                    output_names=['pe_out'],
                    dynamic_axes={'x': {0: 'batch', 1: 'seq_len'}, 'pe_out': {0: 'seq_len'}},
                    opset_version=17,
                    do_constant_folding=True,
                )

                tracer_warnings = [
                    warning for warning in w
                    if 'TracerWarning' in str(warning.category.__name__)
                ]
                if tracer_warnings:
                    warning_messages = [str(warning.message) for warning in tracer_warnings]
                    pytest.fail(f"TracerWarnings in PE export:\n" + "\n".join(warning_messages))

            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            # Test with various offset values and sequence lengths
            for seq_len in [30, 50, 100]:
                for offset_val in [0, 10, 50]:
                    x_test = torch.rand(1, seq_len, config.D_MODEL)
                    offset = torch.tensor(offset_val, dtype=torch.long)

                    with torch.no_grad():
                        pt_out = wrapper(x_test, offset)

                    ort_out = session.run(None, {
                        'x': to_numpy(x_test),
                        'offset': to_numpy(offset)
                    })[0]

                    diff = np.abs(to_numpy(pt_out) - ort_out).max()
                    assert diff < 1e-5, f"PE seq_len={seq_len}, offset={offset_val} max diff: {diff:.6e}"

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


def test_model_dynamo_compatible():
    """Verify torch.compile (Dynamo) works with the model."""
    model = StreamingConformer(num_layers=4)
    model.eval()
    
    states = model.get_initial_states(1, device="cpu")
    x = torch.rand(1, 40, config.N_BINS)
    
    # Compile with Dynamo
    compiled = torch.compile(model, backend="inductor")
    
    # Warmup compilation
    with torch.no_grad():
        (logits, sig_out, bound_out), new_states = compiled(x, states)
    
    # Verify output shapes
    assert logits.shape == (1, 20, config.NUM_CLASSES)
    assert sig_out.shape == (1, 20, config.NUM_SIGNAL_CLASSES)
    assert bound_out.shape == (1, 20, 1)





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
