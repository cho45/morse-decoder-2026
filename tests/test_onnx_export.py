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


class ONNXWrapper(nn.Module):
    """Wrapper to flatten cache states for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = len(model.layers)

    def forward(self, x, pcen_state, sub_cache, *layer_states_flat):
        layer_states = []
        for i in range(0, len(layer_states_flat), 4):
            k = layer_states_flat[i]
            v = layer_states_flat[i+1]
            offset = layer_states_flat[i+2]
            conv = layer_states_flat[i+3]
            layer_states.append(((k, v, offset), conv))

        states = (pcen_state, sub_cache, layer_states)
        (logits, signal_logits, boundary_logits), (new_pcen_state, new_sub_cache, new_layer_states) = self.model(x, states)

        new_states_flat = [new_pcen_state, new_sub_cache]
        for (new_attn, new_conv) in new_layer_states:
            new_states_flat.append(new_attn[0])  # k
            new_states_flat.append(new_attn[1])  # v
            new_states_flat.append(new_attn[2].reshape(()))  # offset
            new_states_flat.append(new_conv)

        return logits, signal_logits, boundary_logits, *new_states_flat


def create_initial_states(batch_size, num_layers, device='cpu'):
    """Create initial cache states with zero values."""
    d_k = config.D_MODEL // config.N_HEAD
    pcen_state = torch.zeros(batch_size, 1, config.N_BINS, device=device)
    sub_cache = torch.zeros(batch_size, 1, 2, config.N_BINS, device=device)

    layer_states_flat = []
    for _ in range(num_layers):
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, 0, d_k, device=device))  # k
        layer_states_flat.append(torch.zeros(batch_size, config.N_HEAD, 0, d_k, device=device))  # v
        layer_states_flat.append(torch.tensor(0, dtype=torch.long, device=device))  # offset
        layer_states_flat.append(torch.zeros(batch_size, config.D_MODEL, config.KERNEL_SIZE - 1, device=device))  # conv

    return pcen_state, sub_cache, layer_states_flat


def export_model_to_onnx(model, onnx_path, seq_len=40):
    """Export model to ONNX format."""
    wrapper = ONNXWrapper(model)
    wrapper.eval()  # Ensure wrapper is in eval mode
    batch_size = 1
    # Use positive inputs for PCEN
    x = torch.rand(batch_size, seq_len, config.N_BINS)
    pcen_state, sub_cache, layer_states_flat = create_initial_states(batch_size, len(model.layers))

    input_names = ['x', 'pcen_state', 'sub_cache']
    output_names = ['logits', 'signal_logits', 'boundary_logits', 'new_pcen_state', 'new_sub_cache']

    dynamic_axes = {
        'x': {0: 'batch_size', 1: 'seq_len'},
        'pcen_state': {0: 'batch_size'},
        'sub_cache': {0: 'batch_size', 2: 'sub_cache_len'},
        'logits': {0: 'batch_size', 1: 'out_seq_len'},
        'signal_logits': {0: 'batch_size', 1: 'out_seq_len'},
        'boundary_logits': {0: 'batch_size', 1: 'out_seq_len'},
        'new_pcen_state': {0: 'batch_size'},
        'new_sub_cache': {0: 'batch_size', 2: 'new_sub_cache_len'},
    }

    for i in range(len(model.layers)):
        input_names.extend([f'attn_k_{i}', f'attn_v_{i}', f'offset_{i}', f'conv_cache_{i}'])
        output_names.extend([f'new_attn_k_{i}', f'new_attn_v_{i}', f'new_offset_{i}', f'new_conv_cache_{i}'])

        dynamic_axes[f'attn_k_{i}'] = {0: 'batch_size', 2: 'attn_cache_len'}
        dynamic_axes[f'attn_v_{i}'] = {0: 'batch_size', 2: 'attn_cache_len'}
        dynamic_axes[f'conv_cache_{i}'] = {0: 'batch_size'}

        dynamic_axes[f'new_attn_k_{i}'] = {0: 'batch_size', 2: 'new_attn_cache_len'}
        dynamic_axes[f'new_attn_v_{i}'] = {0: 'batch_size', 2: 'new_attn_cache_len'}
        dynamic_axes[f'new_conv_cache_{i}'] = {0: 'batch_size'}

    torch.onnx.export(
        wrapper,
        (x, pcen_state, sub_cache, *layer_states_flat),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False  # Use legacy TorchScript exporter for compatibility
    )

    return wrapper, input_names, output_names


def to_numpy(t):
    """Convert tensor to numpy array."""
    return t.detach().cpu().numpy() if t is not None else None


class TestONNXExportNoWarnings:
    """Test that ONNX export produces no TracerWarnings."""

    def test_onnx_export_no_tracer_warnings(self):
        """ONNX export should produce no TracerWarnings."""
        model = StreamingConformer(num_layers=2)
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

    @pytest.fixture
    def model_and_session(self):
        """Create model and ONNX session."""
        model = StreamingConformer(num_layers=2)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        yield model, wrapper, session, input_names, output_names

        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

    @pytest.mark.parametrize("seq_len", [10, 40, 100, 200])
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


class TestONNXStreamingEquivalence:
    """Test streaming inference equivalence between PyTorch and ONNX."""

    @pytest.fixture
    def model_and_session(self):
        """Create model and ONNX session."""
        model = StreamingConformer(num_layers=2)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        yield model, wrapper, session, len(model.layers), output_names

        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

    @pytest.mark.parametrize("chunk_size", [20, 40, 80])
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
        model = StreamingConformer(num_layers=2)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            batch_size = 1
            chunk_size = 40
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
        model = StreamingConformer(num_layers=2)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            wrapper, input_names, output_names = export_model_to_onnx(model, onnx_path)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            batch_size = 1
            chunk_size = 100
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
