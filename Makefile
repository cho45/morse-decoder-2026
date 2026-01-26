.PHONY: test build train onnx test-js evaluate-js

IMAGE_NAME = cw-decoder
DOCKER_RUN = docker run --rm --gpus all -v $(shell pwd):/workspace $(IMAGE_NAME)

# Default to latest checkpoint in checkpoints/ directory
CHECKPOINT ?= $(shell ls -t checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)

build:
	docker build -t $(IMAGE_NAME) .

test:
	$(DOCKER_RUN) python3 -m pytest tests/

train:
	$(DOCKER_RUN) python3 -u train.py $(ARGS)

onnx:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: No checkpoint found in checkpoints/"; \
		echo "Usage: make onnx [CHECKPOINT=path/to/checkpoint.pt]"; \
		exit 1; \
	fi
	@echo "Using checkpoint: $(CHECKPOINT)"
	$(DOCKER_RUN) python3 export_onnx.py --checkpoint $(CHECKPOINT) --output demo/cw_decoder.onnx
	$(DOCKER_RUN) python3 quantize_onnx.py --input demo/cw_decoder.onnx --output demo/cw_decoder_quantized.onnx

performance:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: No checkpoint found in checkpoints/"; \
		exit 1; \
	fi
	@echo "Using checkpoint: $(CHECKPOINT)"
	$(DOCKER_RUN) python3 visualize_snr_performance.py --checkpoint $(CHECKPOINT) --samples 50 --output diagnostics/snr_performance_latest.png

performance_onnx:
	$(DOCKER_RUN) python3 visualize_snr_performance_onnx.py \
		--models demo/cw_decoder.onnx demo/cw_decoder_quantized.onnx \
		--labels "Demo FP32" "Demo INT8" \
		--samples 50 \
		--output diagnostics/snr_performance_onnx_comparison.png

test-js:
	cd demo && npm test

evaluate-js:
	node demo/evaluate_snr.js --model demo/cw_decoder_quantized.onnx