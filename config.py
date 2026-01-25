"""
Global Configuration for CW Decoder.
Centralizing all parameters to ensure consistency across data generation, 
training, and streaming inference.
"""

# Audio / DSP Parameters
SAMPLE_RATE = 16000 # サンプリングレート (Hz)
N_FFT = 400         # FFT窓サイズ (25ms @ 16kHz)
HOP_LENGTH = 160    # フレームの移動間隔 (10ms @ 16kHz)
N_MELS = 16          # メルフィルタバンクの数。700Hz周辺の情報を密に取るため少なめに設定

# Model Architecture Parameters
SUBSAMPLING_RATE = 2 # ConvSubsampling による時間方向の圧縮率
D_MODEL = 256        # Transformer 内の隠れ層の次元数。表現力向上のため再増強。
N_HEAD = 4           # Multi-head Attention のヘッド数。
NUM_LAYERS = 4       # Conformer ブロックの積層数。リズム抽出と言語処理の機能分担を促す。
KERNEL_SIZE = 31     # Depthwise Convolution のカーネルサイズ。約600ms（1文字分）をカバー。
DROPOUT = 0.1        # ドロップアウト率

# Streaming Parameters
MAX_CACHE_LEN = 1000  # ストリーミング推論時の過去キャッシュの最大フレーム数 (約20秒分)
LOOKAHEAD_FRAMES = 30 # 未来の信号をどれだけ参照するか (10フレーム = 100ms)
TARGET_FRAMES = 1000  # 学習時のターゲットフレーム数 (約10秒)

# Phrase Generation Parameters
PHRASE_TEMPLATES = [
    "CQ CQ CQ",
    "CQ DE {call}",
    "QRZ? DE {call}",
    "DE {call} {call}",
    "DE {call} K",
    "TU DE {call}",
    "UR RST {rst}",
    "RST {rst} BK",
    "5NN BK",
    "QRM ES QSB",
    "FB SIGS",
    "QTH {city}",
    "NAME {name}",
    "OP {name}",
    "WX {weather}",
    "TEMP {temp} C",
    "TU FER CALL",
    "GA OM",
    "GM OM",
    "GE OM",
    "PSE K",
    "FB OM TU",
    "HW? BK",
    "QSL? BK",
    "TNX QSO",
    "73 GL SK",
    "73 TU",
    "SEE U AGW",
    "{call} DE {call} CL",
]
COMMON_WEATHER = ["FINE", "RAIN", "CLOUDY", "SNOW", "SUNNY", "HOT", "COLD"]

# Signal Task Parameters
# 0: Background/Padding, 1: Dit, 2: Dah, 3: Intra-char space, 4: Inter-char space, 5: Inter-word space
NUM_SIGNAL_CLASSES = 6

# Vocabulary
# ID 0 は CTC の 'blank' トークンとして予約されているため、文字 ID は 1 から開始する
import string
# 標準的な使用文字。スペースは物理的な「空白」として Signal Head で扱うため、CTC 語彙からは除外する。
STD_CHARS = sorted(list(string.ascii_uppercase + string.digits + "/?.,"))
# 略符号 (Prosigns) やよく使われる略語を独立したトークンとして扱う
PROSIGNS = ["<BT>", "<AR>", "<SK>", "<KA>"]
CHARS = STD_CHARS + PROSIGNS # 全ボキャブラリ
NUM_CLASSES = len(CHARS) + 1 # blank を含めた総クラス数
CHAR_TO_ID = {char: i + 1 for i, char in enumerate(CHARS)} # 文字から ID へのマップ
ID_TO_CHAR = {i + 1: char for i, char in enumerate(CHARS)} # ID から文字へのマップ