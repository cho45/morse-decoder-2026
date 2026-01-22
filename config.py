"""
Global Configuration for CW Decoder.
Centralizing all parameters to ensure consistency across data generation, 
training, and streaming inference.
"""

# Audio / DSP Parameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# Model Architecture Parameters
# Subsampling rate must match plan.md (2x)
SUBSAMPLING_RATE = 2
D_MODEL = 144
N_HEAD = 4
NUM_LAYERS = 6
KERNEL_SIZE = 31
DROPOUT = 0.1

# Streaming Parameters
# 20 seconds at 50Hz (after 2x subsampling from 100Hz/10ms hop)
MAX_CACHE_LEN = 1000

# Vocabulary
# ID 0 is reserved for CTC blank
import string
CHARS = sorted(list(string.ascii_uppercase + string.digits + "/?.,"))
NUM_CLASSES = len(CHARS) + 1
CHAR_TO_ID = {char: i + 1 for i, char in enumerate(CHARS)}
ID_TO_CHAR = {i + 1: char for i, char in enumerate(CHARS)}