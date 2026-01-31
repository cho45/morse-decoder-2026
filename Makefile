.PHONY: test build train onnx test-js evaluate-js show-curriculum

IMAGE_NAME = cw-decoder
DOCKER_RUN = docker run --rm --gpus all --shm-size=2g -v $(shell pwd):/workspace $(IMAGE_NAME)

# Default to latest checkpoint in checkpoints/ directory
CHECKPOINT ?= $(shell ls -t checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)

build:
	docker build -t $(IMAGE_NAME) .

test:
	$(DOCKER_RUN) python3 -m pytest $(if $(ARGS),$(ARGS),tests/)

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
	$(DOCKER_RUN) python3 diagnostics/visualize_snr_performance.py --checkpoint $(CHECKPOINT) --samples 50 --output diagnostics/visualize_snr_performance.png

performance_onnx: onnx
	$(DOCKER_RUN) python3 diagnostics/visualize_snr_performance_onnx.py \
		--models demo/cw_decoder.onnx demo/cw_decoder_quantized.onnx \
		--labels "Demo FP32" "Demo INT8" \
		--samples 50 \
		--output diagnostics/visualize_snr_performance_onnx.png

performance_pt_streaming:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: No checkpoint found in checkpoints/"; \
		exit 1; \
	fi
	@echo "Using checkpoint: $(CHECKPOINT)"
	$(DOCKER_RUN) python3 diagnostics/visualize_snr_performance_pt_streaming.py --checkpoint $(CHECKPOINT) --samples 50 --output diagnostics/visualize_snr_performance_pt_streaming.png

analyze-char-errors:
	$(DOCKER_RUN) python3 -u diagnostics/analyze_char_errors.py --samples 500 --snr "6,-6,-12"

test-js:
	cd demo && npm test

evaluate-js:
	node demo/evaluate_snr.js --model demo/cw_decoder_quantized.onnx

show-curriculum:
	$(DOCKER_RUN) python3 curriculum.py

update-all-diagnostics:
	make performance
	make performance_onnx
	make performance_pt_streaming
	make analyze-char-errors

