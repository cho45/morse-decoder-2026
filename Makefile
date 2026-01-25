.PHONY: test build train onnx

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
	$(DOCKER_RUN) python3 export_onnx.py --checkpoint $(CHECKPOINT)