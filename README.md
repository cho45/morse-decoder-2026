# CW Decoder (Streaming Conformer)

ディープラーニング（Streaming Conformer）を用いた、リアルタイム・モールス信号（CW）復号システムです。
WSL (Ubuntu) 上での学習および推論、最終的には Web ブラウザ上での動作を目指しています。

## 特徴
- **Streaming Conformer:** CTC 損失を用いたストリーミング対応の CNN + Transformer アーキテクチャ。
- **周波数クロッピング:** 広帯域信号からターゲット信号を DSP で特定し、軽量モデルで効率的に復号。
- **堅牢な合成データ:** 人間による打鍵の揺らぎや、HF 帯特有のノイズ・フェージングをシミュレートしたデータで学習。

## 環境構築 (WSL2 / Docker)

NVIDIA Container Toolkit がインストールされた WSL2 環境を前提としています。

### 1. Docker イメージのビルド
```bash
docker build -t cw-decoder .
```

### 2. コンテナの起動（GPU 有効化・ディレクトリマウント）
```bash
docker run --rm -it --gpus all -v `pwd`:/workspace cw-decoder bash
```

### 3. GPU 認識の確認 (コンテナ内)
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. 学習の実行 (コンテナ内)
```bash
python3 train.py --samples-per-epoch 1000 --epochs 10 --batch-size 16
```
ホスト側から直接実行する場合:
```bash
docker run --rm --gpus all -v `pwd`:/workspace cw-decoder python3 train.py --samples-per-epoch 1000 --epochs 3 --batch-size 8
```

## プロジェクト構造
- `data_gen.py`: 高度な人間的揺らぎとノイズをシミュレートするデータジェネレータ。
- `model.py`: Lightweight Streaming Conformer モデルの定義。
- `train.py`: 学習スクリプト。
- `stream_decode.py`: リアルタイム音声入力による推論プロトタイプ。
- `plan.md`: 詳細なプロジェクト計画書。

## 開発フェーズ
1. **フェーズ 1:** データジェネレータの実装
2. **フェーズ 2:** モデル定義と学習（オンザフライ生成）
3. **フェーズ 3:** リアルタイム推論エンジン（Python）の開発
4. **フェーズ 4:** Web ブラウザ（ONNX Runtime Web / WASM）へのデプロイ