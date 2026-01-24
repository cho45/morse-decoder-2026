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
D_MODEL = 64        # Transformer 内の隠れ層の次元数
N_HEAD = 2           # Multi-head Attention のヘッド数
NUM_LAYERS = 2       # Conformer ブロックの積層数
KERNEL_SIZE = 11     # Depthwise Convolution のカーネルサイズ。約220msのコンテキストをカバー
DROPOUT = 0.1        # ドロップアウト率

# Streaming Parameters
MAX_CACHE_LEN = 1000  # ストリーミング推論時の過去キャッシュの最大フレーム数 (約20秒分)
LOOKAHEAD_FRAMES = 30 # 未来の信号をどれだけ参照するか (10フレーム = 100ms)

# Vocabulary
# ID 0 は CTC の 'blank' トークンとして予約されているため、文字 ID は 1 から開始する
import string
# 標準的な使用文字。スペースは CTC のアライメント競合を避けるため除外。
# 単語区切りは、デコード時に文字間の時間的なギャップに基づいて動的に再構築される。
STD_CHARS = sorted(list(string.ascii_uppercase + string.digits + "/?.,"))
# 略符号 (Prosigns) やよく使われる略語を独立したトークンとして扱う
PROSIGNS = ["<BT>", "<AR>", "<SK>", "<KA>", "CQ", "DE"]
CHARS = STD_CHARS + PROSIGNS # 全ボキャブラリ
NUM_CLASSES = len(CHARS) + 1 # blank を含めた総クラス数
CHAR_TO_ID = {char: i + 1 for i, char in enumerate(CHARS)} # 文字から ID へのマップ
ID_TO_CHAR = {i + 1: char for i, char in enumerate(CHARS)} # ID から文字へのマップ