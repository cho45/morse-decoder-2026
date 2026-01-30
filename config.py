"""
Global Configuration for CW Decoder.
Centralizing all parameters to ensure consistency across data generation, 
training, and streaming inference.
"""

# Audio / DSP Parameters
SAMPLE_RATE = 16000 # サンプリングレート (Hz)
N_FFT = 512         # FFT窓サイズ (32ms @ 16kHz)
HOP_LENGTH = 160    # フレームの移動間隔 (10ms @ 16kHz)
F_MIN = 500.0       # 抽出する周波数範囲の下限 (Hz)
F_MAX = 900.0       # 抽出する周波数範囲の上限 (Hz)
# 16000 / 512 = 31.25 Hz/bin
# 500 / 31.25 = 16 (Bin Index)
# 900 / 31.25 = 28.8 -> 29 (Bin Index)
# 29 - 16 + 1 = 14 bins
N_BINS = 14         # スペクトログラムから抽出するビン数
MIN_FREQ = 600.0    # 学習データの最小周波数 (Hz)
MAX_FREQ = 800.0    # 学習データの最大周波数 (Hz)
SNR_REF_BW = 2500.0 # SNR の基準帯域幅 (Hz)。2500Hz は SSB 帯域相当。

# Evaluation Parameters
EVAL_SNR_MIN = -18
EVAL_SNR_MAX = 0
EVAL_SNR_STEP = 1

# Augmentation Parameters
QRN_PROB = 0.3          # 雷ノイズ (Static Crashes) の発生確率
AGC_PROB = 0.5          # AGC シミュレーションの適用確率
DRIFT_PROB = 0.3        # 周波数ドリフトの発生確率
MULTIPATH_PROB = 0.2    # マルチパス/エコーの発生確率
CLIPPING_PROB = 0.2     # クリッピングの発生確率

# Model Architecture Parameters
SUBSAMPLING_RATE = 2 # ConvSubsampling による時間方向の圧縮率
D_MODEL = 256        # Transformer 内の隠れ層の次元数。表現力向上のため再増強。
N_HEAD = 4           # Multi-head Attention のヘッド数。
NUM_LAYERS = 4       # Conformer ブロックの積層数。リズム抽出と言語処理の機能分担を促す。
KERNEL_SIZE = 31     # Depthwise Convolution のカーネルサイズ。約600ms（1文字分）をカバー。
DROPOUT = 0.1        # ドロップアウト率

# Streaming Parameters
MAX_CACHE_LEN = 1000  # ストリーミング推論時の過去キャッシュの最大フレーム数 (約20秒分)
LOOKAHEAD_FRAMES = 10 # 未来の信号をどれだけ参照するか (10フレーム = 100ms)
TARGET_FRAMES = 1000  # 学習時のターゲットフレーム数 (約10秒)

# Phrase Generation Parameters
PHRASE_TEMPLATES = [
    "CQ CQ CQ",
    "CQ DE {call}",
    "CQ TEST DE {call}",
    "QRZ? DE {call}",
    "DE {call} {call}",
    "DE {call} K",
    "TU DE {call}",
    "UR RST {rst}",
    "RST {rst} BK",
    "R {rst} K",
    "R {rst} TU",
    "5NN BK",
    "QRM ES QSB",
    "FB SIGS",
    "QTH {city} {city}",
    "NAME {name} {name}",
    "OP {name} {name}",
    "WX {weather}",
    "TEMP {temp} C",
    "TU FER CALL",
    "PSE K",
    "FB OM TU",
    "HW? BK",
    "QSL? BK",
    "TNX QSO",
    "TNX FER UR CALL",
    "73 GL SK",
    "73 TU E E",
    "SEE U AGW",
    "{call1} DE {call2} K",
    "{call1} DE {call2} GM",
    "{call1} DE {call2} GA",
    "{call1} DE {call2} GE",
    "{call1} DE {call2} CL",
]
COMMON_WEATHER = ["FINE", "RAIN", "CLOUDY", "SNOW", "SUNNY", "HOT", "COLD"]

# Signal Task Parameters
# 0: Background/Padding/Space, 1: Dit, 2: Dah, 3: Inter-word space
NUM_SIGNAL_CLASSES = 4

# Vocabulary
# ID 0 は CTC の 'blank' トークンとして予約されているため、文字 ID は 1 から開始する
import string
# 標準的な使用文字。スペースは物理的な「空白」として Signal Head で扱うため、CTC 語彙からは除外する。
# [修正] MORSE_DICT に存在するすべての記号を網羅するように修正。
# 不足していたもの: ' ! & : ; = + _ " $ @
STD_CHARS = sorted(list(string.ascii_uppercase + string.digits + "/?.,-()'!&:;=+_\"$@"))
# 略符号 (Prosigns) やよく使われる略語を独立したトークンとして扱う
# [修正] <AA> を追加し、<HHHH> を <HH> に修正。
PROSIGNS = ["<NJ>", "<DDD>", "<SK>", "<KA>", "<SOS>", "<VE>", "<HH>", "<AA>"]
CHARS = STD_CHARS + PROSIGNS # 全ボキャブラリ
NUM_CLASSES = len(CHARS) + 1 # blank を含めた総クラス数
CHAR_TO_ID = {char: i + 1 for i, char in enumerate(CHARS)} # 文字から ID へのマップ
ID_TO_CHAR = {i + 1: char for i, char in enumerate(CHARS)} # ID から文字へのマップ
