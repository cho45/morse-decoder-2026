.PHONY: test build

IMAGE_NAME = cw-decoder
DOCKER_RUN = docker run --rm --gpus all -v $(shell pwd):/workspace $(IMAGE_NAME)

build:
	docker build -t $(IMAGE_NAME) .

test:
	$(DOCKER_RUN) python3 -m pytest tests/