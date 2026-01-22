# Use official PyTorch image which has torchaudio pre-installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

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
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["bash"]