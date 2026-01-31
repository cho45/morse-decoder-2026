# Use official PyTorch image which has torchaudio pre-installed
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies for PyAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Default command
CMD ["bash"]