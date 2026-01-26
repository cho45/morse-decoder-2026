import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import argparse

def quantize_model(input_path, output_path):
    print(f"Quantizing model: {input_path} -> {output_path}")
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input model not found at {input_path}")
        return

    # Perform dynamic quantization
    # We target Linear layers for INT8 quantization
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    # Compare sizes
    initial_size = os.path.getsize(input_path) / (1024 * 1024)
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Initial model size: {initial_size:.2f} MB")
    print(f"Quantized model size: {final_size:.2f} MB")
    print(f"Reduction: {(1 - final_size/initial_size)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 with pre-processing")
    parser.add_argument("--input", type=str, default="demo/cw_decoder.onnx", help="Input ONNX model path")
    parser.add_argument("--output", type=str, default="demo/cw_decoder_quantized.onnx", help="Output quantized ONNX model path")
    args = parser.parse_args()

    quantize_model(args.input, args.output)