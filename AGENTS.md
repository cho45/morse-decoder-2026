# Agent Interaction Rules

**CRITICAL:**
README.md を読まずに勝手な推論や実行を行うエージェントは、プロジェクトの整合性を損なうため厳禁です。
必ず README.md を読み、プロジェクトの文脈と手順を理解した上で行動してください。

## テストの実行方法
ホスト環境で直接 `pytest` を実行してはいけません。必ず Docker コンテナを使用して実行してください。
`Makefile` が用意されているため、以下のコマンドで実行可能です。

### 全テスト実行
```bash
make test
# または
docker run --rm --gpus all -v `pwd`:/workspace cw-decoder python3 -m pytest tests/
```

### 単一テスト実行
```bash
# 特定のテストファイルのみ
make test ARGS="tests/test_model.py"
# または
docker run --rm --gpus all -v `pwd`:/workspace cw-decoder python3 -m pytest tests/test_model.py

# 特定のテスト関数のみ
docker run --rm --gpus all -v `pwd`:/workspace cw-decoder python3 -m pytest tests/test_model.py::test_conv_subsampling_consistency

# 特定のパラメータのみ
docker run --rm --gpus all -v `pwd`:/workspace cw-decoder python3 -m pytest tests/test_model.py::test_model_consistency[40]
```

### JavaScript テスト実行
```bash
cd demo && npm test
```

## コードスタイルガイドライン

### Python
#### インポート順序
1. 標準ライブラリ（os, sys, math, random, etc.）
2. サードパーティライブラリ（torch, numpy, scipy, etc.）
3. ローカルモジュール（import config, from model import ...）

```python
import os
import sys
import math

import torch
import numpy as np
import scipy.signal

import config
from model import StreamingConformer
from data_gen import CWDataset
```

#### 型アノテーション
すべての関数には型ヒントを必須：
```python
def forward(self, x: torch.Tensor, 
            cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

#### 命名規約
- クラス: PascalCase（`StreamingConformer`, `MorseGenerator`）
- 関数・変数: snake_case（`compute_spec_frames`, `waveform`）
- 定数: UPPER_CASE（`SAMPLE_RATE`, `N_BINS`, `D_MODEL`）
- プライベートメソッド: 前にアンダースコア（`_compute_cache`）

#### コンフィギュレーション
すべての定数は `config.py` に集中させる。新しく定数を追加する際は必ず `config.py` に追加し、他ファイルから import する。

#### エラーハンドリング
- 予期されるエラー: 適切な例外をキャッチし、意味のあるエラーメッセージを出力
- 予期せぬエラー: ログに記録して再送
- ONNX 互換性のため、TorchScript tracing に対応する記述を優先

#### ドキュメント
- 関数・クラスには簡潔な docstring（日本語可）
- 複雑なアルゴリズムにはステップごとのコメント
- TODO/FIXME は具体的かつアクション可能に

### JavaScript (demo/)
#### モジュール
ES6 modules を使用：
```javascript
import { DSP } from './dsp.js';
export function computeSpecFrame(samples, sampleRate) { ... }
```

#### 命名規約
- 関数・変数: camelCase（`computeSpecFrame`, `spectrogram`）
- 定数: UPPER_CASE（`N_BINS`, `SAMPLE_RATE`）
- クラス: PascalCase（`DSP`, `CTCDecoder`）

#### 型情報（JSDoc）
重要な関数には JSDoc を記述：
```javascript
/**
 * Compute all spectrogram frames for a given waveform.
 * @param {Float32Array} waveform - Audio samples
 * @param {number} sampleRate - Current audio sample rate
 * @returns {Float32Array} Flat array of spectrogram frames [T, N_BINS]
 */
```

#### 数値演算
- 浮動小数点精度を考慮し、IIR フィルタ内部状態は Float64Array 使用
- 結果は必要に応じて Float32Array に変換

### テストの書き方
#### pytest
- `pytest.mark.parametrize` を活用して境界条件を網羅
- ストリーミング処理の一貫性を検証するテストを重視
- テスト名は具体的な挙動を表す名前に（`test_conv_subsampling_consistency`）

#### Red/Green 原則
バグ修正時は必ず失敗するテストを先に書き、修正後にパスさせること。

### プロジェクト特有の注意点
1. **Docker 必須**: すべての Python コード実行は Docker 内で行う
2. **ONNX 互換性**: モデル定義は ONNX export に対応する記述必須
3. **ストリーミング性**: 全ての処理はリアルタイム性を考慮し、因果律を守る
4. **マルチタスク学習**: CTC Head, Signal Head, Boundary Head の3つを統合
5. **日本語ドキュメント**: README.md 等主要なドキュメントは日本語