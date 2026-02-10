# Data Generation Module (`data_gen.py`)

モールス信号の学習データを生成するためのモジュール。
1サンプル単位の精度（Sample-level accuracy）と、信号とラベルの数学的一致を保証することを目的とする。

## 設計思想

- **Single Source of Truth (SSoT)**: サンプル数ベースの `timing` リストを全工程の絶対的な基準とする。
- **Integer Precision**: 時間管理はすべて整数（サンプル数）で行い、浮動小数点による累積誤差（Drift）を排除する。
- **Atomic Character Addition**: 文字を生成バッファに追加する際、「文字信号 ＋ 続く空白」の両方が収まる場合のみ追加を許可し、ラベルの不整合（尻切れトンボ）を防止する。

---

## 用語とデータ構造の定義

| 用語 | データ型 | 説明 | 具体例 |
| :--- | :--- | :--- | :--- |
| **`text`** | `str` | 人間が読める平文テキスト。 | `"PARIS"` |
| **`tokens`** | `List[int]` | `MorseEncoder` が出力する抽象的な信号要素の列。文字内の要素（Dit/Dah）と要素間のギャップのみを含む。 | `[1, 0, 2]` (A) |
| **`timing`** | `List[Tuple[int, int]]` | **SSoT (単一の真実)**。各要素のクラスIDと、その継続サンプル数のペアのリスト。 | `[(1, 480), (4, 1440)]` |
| **`waveform`** | `np.ndarray` | `WaveformGenerator` が生成するオーディオ波形データ。振幅は -1.0 〜 1.0。 | `[0.0, 0.1, ...]` |
| **`signal_labels`**| `np.ndarray` | フレーム単位の信号クラスラベル。各フレームがどの状態（無音/短点/長点）にあるかを示す。 | `[0, 1, 1, 0, ...]` |
| **`boundary_labels`**| `np.ndarray` | フレーム単位のバウンダリラベル。文字が終了した直後のフレームを 1、それ以外を 0 とする。 | `[0, 0, 1, 0, ...]` |

### クラスID定義

#### 内部タイミング管理用 (`timing` リスト内)
モデルが学習する信号クラスと衝突しないよう、一部に負数を採用しています。

- `SIG_ID_DIT (1)`: 短点 (.)
- `SIG_ID_DAH (2)`: 長点 (-)
- `SIG_ID_WORD_SPACE (3)`: 単語間空白 (7 units)
- `INTERNAL_ID_INTRA_GAP (-1)`: 文字内の要素間ギャップ (1 unit)
- `INTERNAL_ID_INTER_CHAR (-2)`: 文字間ギャップ (3 units)
- `SIG_ID_BLANK (0)`: パディングやプリ/ポストサイレンス

#### 学習ターゲットラベル (`signal_labels` 内)
`LabelGenerator` により以下の 4 クラスに集約されます。

- `0`: 背景 (Background / Padding / Silence / Intra-char gap / Inter-char gap) -> SIG_ID_BLANK
- `1`: 短点 (Dit)
- `2`: Dah (長点)
- `3`: 単語間空白 (Inter-word space)

> **Note**: 学習時の `SignalHead` はこれら 4 クラスを予測するように構成されています。

---

## クラス構成

### 1. `MorseEncoder`
テキストとトークン（Dit, Dah, Gap）の相互変換を担当。

- **責務**:
    - 文字列からモールス符号トークンへのエンコード。
    - トークンからテキストへのデコード（検証用）。
- **主要メソッド**:
    - `text_to_tokens(text)`: 文字列を `[1, 3, 2, 0, ...]` 形式のトークン列に変換。
    - `tokens_to_text(tokens)`: トークン列を文字列に復元。
- **トークン定義**:
    - `0`: Gap (Intra-character)
    - `1`: Dit
    - `2`: Dah

### 2. `TimingGenerator`
トークン列を具体的なサンプル数のシーケンスに変換。**このモジュールが SSoT を提供する。**

- **責務**:
    - WPM/Farnsworth WPM に基づくサンプル数計算。
    - **人間らしいリズムの揺らぎ (Human Artifacts)** の適用:
        - `jitter`: 個々の Dit/Dah の長さをランダムに変動させ、手打ちの揺れを再現。
        - `weight`: 全体の信号長を一律に増減させ、個人の癖（重い/軽い打ち方）を再現。
    - バッファサイズ制限に基づく生成の切り出し。
    - 「文字＋空白」の完全性（Atomic Addition）の管理。
- **入力**:
    - `tokens`: `MorseEncoder` から出力されたトークン列。
    - `max_duration`: 生成する最大サンプル数。
- **出力**:
    - `timing`: `List[Tuple[class_id, num_samples]]`
- **出力クラス定義**:
    - `INTERNAL_ID_GAP (-3)`: 長い無音区間
    - `INTERNAL_ID_INTER_CHAR (-2)`: 文字間ギャップ
    - `INTERNAL_ID_INTRA_GAP (-1)`: 文字内の要素間ギャップ
    - `SIG_ID_BLANK (0)`: パディング等の無音
    - `SIG_ID_DIT (1)`: 短点
    - `SIG_ID_DAH (2)`: 長点
    - `SIG_ID_WORD_SPACE (3)`: 単語間空白

### 3. `WaveformGenerator`
`timing` リストからオーディオ波形を生成。

- **責務**:
    - 指定されたサンプル数に基づくサイン波の生成。
    - 振幅（Gain）の適用とノイズ・フェード処理。
- **入力**:
    - `timing`: `TimingGenerator` の出力。
- **出力**:
    - `waveform`: `np.ndarray` (float32, [-1.0, 1.0])

### 4. `LabelGenerator`
`timing` リストからフレーム単位の学習ラベルを生成。

- **責務**:
    - `timing`（サンプル単位）を `HOP_LENGTH` で割り、フレーム単位のラベルへ変換。
    - 信号ラベル（4クラス：Silence, Dit, Dah, WordSpace）の生成。
    - バウンダリラベル（Binary：文字の終端フレーム）の生成。
- **入力**:
    - `timing`: `TimingGenerator` の出力。
    - `num_frames`: 期待される総フレーム数。
- **出力**:
    - `signal_labels`: `np.ndarray` (shape: [frames])
    - `boundary_labels`: `np.ndarray` (shape: [frames], 0 or 1)

### 5. `MorseGenerator`
上記クラス群をオーケストレートするファサードクラス。

- **責務**:
    - テキスト、WPM、SNR などのパラメータから、最終的な学習サンプルを一括生成。
- **主要メソッド**:
    - `estimate_duration(text, wpm, ...)`: `TimingGenerator` を使用して、必要なサンプル数を正確に見積もる。

---

## 時間制限と固定長化の責務

モデルの学習には固定長（`config.TRAIN_DURATION`, デフォルト 10秒）のデータが必要ですが、本モジュールでは以下のステップで段階的に制限を適用しています。

1.  **事前見積もり (`MorseGenerator`)**:
    指定された WPM と `max_duration` から、収まるはずの最大文字数を算出し、生成するテキストの長さを制限する。
    - **フレーズ生成時 (Adaptive WPM)**: 1〜2個のフレーズを連結。2個の場合は間に GAP を挿入する。フレーズ全体が `max_duration` に収まるよう、**GAPの固定時間を差し引いた正味の時間**に基づいて WPM を動的に調整する。
    - **ランダム生成時 (WPM優先)**: 指定された WPM を優先し、その速度で収まる最大文字数を算出してテキストを制限する。また、一定確率（`gap_prob`）で文字列内のランダムな位置に 1.0秒以上の GAP を1箇所挿入する。
2.  **厳密な切り出し (`TimingGenerator`)**:
    `generate_timing` 内で、サンプル数単位で残り時間を監視。文字信号とその後の空白（文字間または単語間）の両方が `max_duration` 内に完結する場合のみ追加を許可し、それ以外は**原子的に切り捨てる**。
3.  **固定長化とパディング (`generate_sample`)**:
    切り出された結果が `max_duration` に満たない場合、末尾を `SIG_ID_BLANK` でパディングし、常に正確に `max_duration` 秒分のテンソルを出力する。

### 6. `HFChannelSimulator`
クリーンな信号に対して、現実の短波通信（HF）環境における物理的な歪みを付加する。

- **責務**:
    - 電離層反射によるフェーディング（マルチパス）のシミュレート。
    - 各種ノイズ（白色雑音、雷ノイズ QRN、混信 QRM）の混入。
    - 周波数ドリフトや帯域制限（フィルター）の適用。
- **シミュレート項目**:
    - **Fading**: `Rayleigh` / `Rician` モデルに基づく振幅変動。
    - **AWGN**: 加法的白色ガウス雑音（SNR 指定に基づく）。
    - **QRN**: 突発的なパルス性ノイズ（Static Crashes）。
    - **QRM**: 近接周波数での他のモールス信号やノイズによる干渉。
    - **Frequency Drift**: 送受信機の不安定性によるピッチ変動。

---

## 信号生成パイプライン

本モジュールは、**「理想的な信号の生成（Source）」**と**「通信路のシミュレーション（Channel）」**を明確に分離して設計されています。

### 1. Source Generation (`MorseGenerator`)
理想的なモールス信号と、それに対応する正解ラベルを生成する。

- **責務**: SSoT（`TimingGenerator`）に基づき、1サンプルの狂いもない波形とラベルのセットを作成する。
- **構成要素**: `MorseEncoder`, `TimingGenerator`, `WaveformGenerator`, `LabelGenerator`
- **出力**: クリーンな波形、信号ラベル、バウンダリラベル。
- **特徴**: 通信路のノイズや歪みは一切含まない。

### 2. Channel Simulation (`generate_sample`)
`MorseGenerator` が作ったクリーンな信号に対し、物理的な歪みを加えて現実の無線信号（HF通信）をシミュレートする。

- **責務**: 学習モデルがノイズやフェーディングに耐えられるよう、信号を劣化させる。
- **構成要素**: `HFChannelSimulator` (ノイズ、帯域制限、フェーディング)、ゲイン調整、正規化。
- **出力**: 通信路の歪みが加わった波形、正解ラベル、最終的なテキスト。

### 3. Data Loading (`CWDataset`)
PyTorch の学習ループにデータを供給する。

- **責務**:
    - `generate_sample` を繰り返し呼び出し、設定されたカリキュラム（WPM/SNR等の範囲）に沿った多様なサンプルをバッチとして提供する。
    - **多様性の付加**: 1〜2個のフレーズ連結、およびランダムな長い無音（GAP）の挿入を制御する。

---
