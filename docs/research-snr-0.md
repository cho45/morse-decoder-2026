## モールス信号対ノイズ評価とML

# ノイズ環境下におけるモールス信号聴覚受信の定量的評価と機械学習モデルのための標準化された評価指標に関する包括的研究報告書

## 概要

本報告書は、無線通信の黎明期から現代に至るまで使用され続けているモールス信号（CW: Continuous Wave）の受信性能評価について、人間による聴覚受信の心理音響学的特性と、近年急速に発展している深層学習（Deep Learning）を用いた自動復号モデルの評価指標との間の乖離を分析し、両者を統一的に比較評価するための枠組みを提言するものである。特に、機械学習分野の研究において頻繁に見過ごされている「帯域幅（Bandwidth）」の概念に着目し、ITU-R（国際電気通信連合無線通信部門）の勧告や熟練オペレーターによる微弱信号受信能力の研究（W2RSレポート等）に基づいた**帯域固定SNR（Fixed-Bandwidth SNR）**の定義を厳密に定式化する。さらに、評価指標として単なる正解率（Accuracy）ではなく、レーベンシュタイン距離に基づく**文字誤り率（CER: Character Error Rate）**の採用と、フェージングやインパルスノイズを含んだより実践的なデータセット構築の必要性を論じる。本稿は、通信工学、音響心理学、機械学習の知見を融合し、次世代の自動モールス復号器が人間の「カクテルパーティー効果」に匹敵、あるいは凌駕するための定量的基盤を提供する。

## 1. 序論：アナログ知覚とデジタル評価の断絶

モールス信号は、最も原始的なデジタル通信形式でありながら、その復号プロセスは長らく人間の高度な聴覚・認知能力に依存してきた。熟練した無線通信士は、激しい雑音や混信（QRM）、信号強度の変動（QSB: フェージング）の中から、特定の周波数のトーンだけを「脳内フィルタ」で抽出し、文脈（コンテキスト）を補完しながらメッセージを再構成する能力を持つ。

一方で、近年の機械学習（ML）、特にディープニューラルネットワーク（DNN）の発展により、このタスクをアルゴリズムで代替しようとする試みが活発化している。MorseNet <sup>[[1](https://ieeexplore.ieee.org/iel7/6287639/8948470/09183940.pdf)]</sup> や LSTM-CTC <sup>[[2]](https://github.com/MaorAssayag/morse-deep-learning-detect-and-decode)</sup> といったモデルは、従来の手法を凌駕する性能を示しているが、その評価手法には重大な欠陥が存在する。それは、「ノイズ」の定義における物理的・工学的な不整合である。

### 1.1 研究の背景と問題の所在

通信工学の世界では、信号対雑音比（SNR）は常に「帯域幅」とセットで語られる。しかし、機械学習の文脈では、SNRはしばしば離散的なサンプリングデータ上のエネルギー比として計算され、物理的な帯域幅への正規化が行われないまま報告されるケースが散見される <sup>[[3]](https://www.wavewalkerdsp.com/2024/07/01/calculate-signal-to-noise-ratio-snr-in-simulation/)</sup>。

例えば、サンプリング周波数が 8kHz のデータセットにおける「SNR -5dB」と、アマチュア無線家が語る「SSBフィルター（2.5kHz）を通した際の SNR -5dB」は、全く異なる物理的状況を指す。この定義の揺らぎは、人間とAIの性能を直接比較することを不可能にし、実用的なアプリケーション（自動受信システム、支援技術等）の開発を阻害している。

### 1.2 本報告書の構成

本報告書は以下の構成で、この断絶を埋めるための包括的なリサーチ結果を提供する。

1. **CW受信の物理学とSNRの定義**: 帯域幅とノイズ電力の関係を数理的に整理し、通信業界標準である「2500Hz参照帯域」の根拠を示す。
2. **人間による聴覚受信の限界（定量的評価）**: W2RSの研究やITU-R勧告に基づき、人間がどの程度のSNRで通信可能かを定量化する。
3. **機械学習モデルの現状と課題**: 最新のDNNモデルの構造と、現在用いられている評価指標の問題点を分析する。
4. **標準化された評価指標の提案**: 機械学習モデルの評価において、物理的な実環境と整合性を保つための「帯域固定SNR」の計算手法と、適切な精度指標（CER等）を提言する。

## 2. CW受信におけるSNRと帯域幅の物理学的基礎

モールス信号の対ノイズ性能を定量的に議論するためには、まず信号（Signal）と雑音（Noise）、そして帯域幅（Bandwidth）の相互関係を物理学的に定義する必要がある。CWは連続波（Continuous Wave）をオン・オフ変調（OOK）したものであり、その占有帯域幅は極めて狭いことが特徴である。

### 2.1 ノイズ電力と帯域幅の関係性

自然界や受信機内部で発生する熱雑音（ホワイトノイズ）は、周波数スペクトル上で平坦に分布していると仮定される（AWGN: Additive White Gaussian Noise）。このとき、受信機が拾うノイズの総電力  $P_n$  は、受信機のフィルタ帯域幅  $B$  に比例する。

$$ P_n = N_0 \times B $$

ここで、 $N_0$  は雑音電力スペクトル密度（Watts/Hz）である。
一方、CW信号の電力  $S$  は、信号がフィルタの通過帯域内に収まっている限り、フィルタの帯域幅を狭めても減衰しない（理想的な場合）。

したがって、SNR（信号対雑音比）は以下の式で表される。

$$ SNR = \frac{S}{N_0 \times B} $$

この式が示唆する事実は極めて重要である。**「帯域幅  $B$  を半分にすれば、ノイズ電力は半分になり、SNRは 3dB 向上する」**ということである <sup>[[5]](https://ham.stackexchange.com/questions/15886/calculating-the-signal-to-noise-ratio-for-cw-morse-code-signals)</sup>。

#### 2.1.1 帯域幅によるSNRの変化の実例

ある一定の信号強度とノイズ環境下において、受信フィルタを切り替えた場合のSNRの変化を以下に示す <sup>[[6]](https://kf6hi.net/radio/SNR.html)</sup>。

- **2500 Hz フィルタ（SSB標準）**: 基準 SNR =  $X$  dB
- **500 Hz フィルタ（CW標準）**:  $X + 10 \log_{10}(2500/500) = X + 7$  dB
- **100 Hz フィルタ（狭帯域DSP/聴覚フィルタ）**:  $X + 10 \log_{10}(2500/100) = X + 14$  dB

つまり、2500Hzの帯域で測定して「-10dB（ノイズに埋もれている）」と評価される信号であっても、100Hzのフィルタを通せば「+4dB（信号がノイズより強い）」となり、容易に検知可能となる。これが、CWが「微弱信号に強い」とされる物理的な理由である。

### 2.2 通信業界における標準参照帯域：2500Hz

定量的な評価を行う際、基準となる「ものさし」が必要である。アマチュア無線や業務無線（HF帯）の世界では、慣習的に **2500Hz**（または2.4kHz、3kHz）がSNRの参照帯域幅として用いられる <sup>[[5]](https://ham.stackexchange.com/questions/15886/calculating-the-signal-to-noise-ratio-for-cw-morse-code-signals)</sup>。

#### 2.2.1 なぜ2500Hzなのか

1. **SSBモードとの互換性**: 現代の無線機の多くはSSB（単側波帯）通信を主体に設計されており、標準的なクリスタルフィルタやDSPフィルタの帯域幅が2.4kHz〜2.7kHzであるため。
2. **Sメーターの校正**: 受信機の信号強度計（Sメーター）やノイズフロアの測定値は、通常この帯域幅での総電力を基準にしている。
3. **比較の公平性**: 異なる変調方式（SSB, CW, FM, デジタルモード）の抗堪性を比較する際、同一のノイズ帯域幅を分母としてSNRを計算することで、変調方式ごとの「プロセスゲイン（帯域圧縮効果）」を含めた性能比較が可能になる <sup>[[7]](https://pa3fwm.nl/technotes/tn09b.html)</sup>。

したがって、機械学習モデルの性能を実環境の指標と照らし合わせる場合も、この**「2500Hz帯域におけるSNR」に換算して評価する**ことが、最も整合性の取れたアプローチとなる。

## 3. 人間による聴覚受信能力の定量的評価

機械学習モデルの目標値（Ground Truth）として、人間がどの程度のノイズ環境下でモールス信号を解読できるかを知ることは不可欠である。人間の聴覚系は、非線形かつ適応的なフィルタリング機能を持っており、その性能は驚くほど高い。

### 3.1 「脳内フィルタ」と臨界帯域

レイ・ソイファー（W2RS）による著名な研究「The Weak-Signal Capability of the Human Ear（人間の耳の微弱信号受信能力）」 <sup>[[6]](https://kf6hi.net/radio/SNR.html)</sup> によれば、熟練したCWオペレーターの聴覚・脳機能は、可変帯域フィルタとして機能する。

- **集中時の実効帯域幅**: 人間が特定のトーン（例：600Hz）に集中してモールス信号を聞き取ろうとする際、脳内の処理帯域幅は約 **50Hz 〜 100Hz** まで狭まると推定されている <sup>[[9]](http://www.g1ogy.com/www.n1bug.net/tech/w2rs/The%20Human%20Ear.pdf)</sup>。
- **カクテルパーティー効果**: 広帯域のノイズの中に埋もれた信号であっても、周波数（ピッチ）とリズムの特徴を手がかりに、信号成分だけを「浮き上がらせて」知覚することができる。

### 3.2 W2RSによる「ZROテスト」の分析結果

1980年代から90年代にかけてAMSAT（アマチュア衛星通信協会）が行った「ZROテスト」のデータは、人間の限界性能を示す貴重な資料である <sup>[[9]](http://www.g1ogy.com/www.n1bug.net/tech/w2rs/The%20Human%20Ear.pdf)</sup>。このテストでは、ノイズレベルを固定し、信号レベルを段階的に下げながらランダムな数字列の受信を試みた。

#### 3.2.1 受信限界の定量値

W2RSの分析に基づき、2500Hz帯域幅に換算したSNR限界値を以下に示す。

| 受信状況 | 100Hz帯域でのSNR | 2500Hz帯域換算SNR | 状態記述 |
| --- | --- | --- | --- |
| 完全なコピー | +0 dB | -14 dB | ほぼエラーなしで受信可能 |
| 50% コピー | -3.6 dB | -17.6 dB | 熟練者が数字列を半分程度判読可能 |
| 信号存在の検知 | -6.6 dB | -20.6 dB | 何か鳴っていることは分かるが内容は不明 |

**重要な洞察**: 熟練した人間は、2500Hzの帯域においてノイズ電力が信号電力の約60倍（+18dB）もある状況下（SNR -18dB）で、内容を解読できる能力を持っている。これは、機械学習モデルが目指すべき「人間超え（Super-human）」のベンチマークとなる数値である。

### 3.3 ITU-R F.339 勧告における基準

一方、国際的な通信規格である ITU-R Recommendation F.339 <sup>[[12]](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-8-201302-I!!PDF-E.pdf)</sup> は、商業通信としての信頼性を重視したより保守的な値を規定している。

**表1: ITU-R F.339-8 におけるモールス電信（A1A/A1B）の所要SNR**
*(安定した伝搬条件における値)*

| 通信方式 | 速度 (Baud) | 受信帯域幅 (Hz) | 所要SNR (dB) | 通信品質グレード |
| --- | --- | --- | --- | --- |
| A1A (聴覚受信) | 8 Bd (~10 WPM) | 3000 | -6 dB | Just Usable（かろうじて使用可能） |
| A1A (聴覚受信) | 8 Bd (~10 WPM) | 3000 | +3 dB | Marginally Commercial（商業利用限界） |
| A1B (機械受信) | 50 Bd (プリンタ) | 250 | +4 dB | ビット誤り率 10−3 |

*注: ここでのSNRは、指定された帯域幅（3000Hzまたは250）内での電力比である。*

**分析**:

- ITUの「Just Usable（-6 dB @ 3kHz）」は、W2RSの限界値（-18 dB @ 2.5kHz）と比較して **12 dB** も高い（甘い）基準となっている。
- これは、ITU基準が「一般のオペレーターが疲労なく業務を行えるレベル」あるいは「確実な通信」を想定しているのに対し、W2RSは「極限状態でのDX（遠距離）通信」を想定しているためである。
- 機械学習モデルの実用性を評価する場合、まずはITU基準（-6 dB）のクリアを目指し、最終的にはW2RS基準（-18 dB）への挑戦がロードマップとなる。

### 3.4 誤り率とSNRの相関曲線（Waterfall Curve）

人間の受信性能は、ある閾値を境に急激に悪化する「ウォーターフォール特性」を示す。

- SNR > -10 dB (2.5kHz ref): ほぼ100%の正解率。
- SNR -10 dB 〜 -15 dB: 正解率が緩やかに低下（コンテキスト補完が効く領域）。
- SNR < -18 dB: 急激に解読不能になる（クリフ効果） <sup>[[13]](https://la3za.blogspot.com/2013/10/studies-on-morse-code-recognition.html)</sup>。

## 4. 機械学習モデルによるモールス復号の現状

従来のDSP（デジタル信号処理）による復号器（GoertzelアルゴリズムやPLLを用いたトーン検出器）は、SNRが高い環境では有効だが、ノイズや混信に弱いという欠点があった。これに対し、近年の深層学習モデルは「パターン認識」のアプローチをとることで、耐ノイズ性能を飛躍的に向上させている。

### 4.1 代表的なモデルアーキテクチャ

#### 4.1.1 MorseNet (CNN + BiLSTM)

Liらによって提案された **MorseNet** <sup>[[1]](https://ieeexplore.ieee.org/iel7/6287639/8948470/09183940.pdf)</sup> は、現在最先端（State-of-the-Art）とされるモデルの一つである。

- **入力**: 信号のスペクトログラム（時間-周波数画像）。
- **構造**:

  - **CNN (Convolutional Neural Network)**: スペクトログラムから特徴量（信号の「線」の形状）を抽出する。画像認識の技術を応用し、ノイズという「背景」から信号という「物体」を検出する。
  - **BiLSTM (Bidirectional Long Short-Term Memory)**: 時間的なシーケンス情報を前後双方向から解析し、ドット・ダッシュのパターンを文字に変換する。
- **強み**: 周波数ドリフト（信号の周波数がふらつく現象）や、近接周波数の干渉（QRM）に対して極めて強い。画像として信号の軌跡を捉えるため、特定の周波数にロックする必要がない。

#### 4.1.2 LSTM-CTC (End-to-End)

Maor Assayagらによる実装 <sup>[[2]](https://github.com/MaorAssayag/morse-deep-learning-detect-and-decode)</sup> は、リカレントニューラルネットワーク（RNN）の一種であるLSTMに、**CTC (Connectionist Temporal Classification)** 損失関数を組み合わせたものである。

- **CTCの利点**: 入力（音声フレーム）と出力（テキスト文字）の長さが一致していなくても学習が可能。モールス信号は速度（WPM）によって1文字あたりの長さが変わるため、CTCによるアライメント不要な学習は非常に有効である。
- **性能**: SNR -5dB（帯域定義は曖昧だが、おそらく広帯域）以上でCER < 2%を達成しているとされる。

### 4.2 データセットの課題

機械学習モデルの訓練には大量のラベル付きデータが必要である。現在、最も広く参照されているのが **Sourya Deyらによる "Morse Code Datasets"** <sup>[[15]](https://souryadey.github.io/research/material/SouryaDey_ICCCNT2018_Presentation.pdf)</sup> である。

- **生成方法**: Pythonスクリプトによる合成音声。AWGN（加法性白色ガウス雑音）を付加。
- **サンプリングレート**: 8kHz。
- **欠点**:

  - ノイズが単純なAWGNに限られており、実際の無線通信で発生するインパルスノイズ（雷ノイズ）やフェージングが含まれていない。
  - 信号のタイミングが機械的で正確すぎるため、人間が手打ちした際のリズムの揺らぎ（スイング）に対する汎化性能が未知数である。

### 4.3 機械学習におけるSNR定義の落とし穴

ML論文で報告される「SNR」と、通信工学の「SNR」の間には、**帯域幅の正規化**に関する決定的な齟齬がある。多くのML実装（Pythonの `numpy` 等で計算）では、以下のようにSNRを計算する <sup>[[3]](https://www.wavewalkerdsp.com/2024/07/01/calculate-signal-to-noise-ratio-snr-in-simulation/)</sup>。

$$ SNR_{sample} = 10 \log_{10} \left( \frac{\sum S^2}{\sum N^2} \right) $$

この計算におけるノイズ電力  $\sum N^2$  は、**ナイキスト周波数（サンプリングレートの半分）までの全帯域**に含まれるノイズである。
例えば、サンプリングレート  $F_s = 8000$  Hz の場合、ノイズ帯域幅は 4000 Hz となる。一方、 $F_s = 44100$  Hz であれば、帯域幅は 22050 Hz となり、同じ  $N_0$ （雑音密度）であっても総ノイズ電力は約5.5倍（+7.4dB）になる。

**問題点**: 「SNR -5dBで成功した」という報告があっても、そのデータのサンプリングレートやFFT幅が明記され、かつ標準帯域に正規化されていなければ、その数値は**物理的な受信性能指標として無意味**である。

## 5. 機械学習モデルの評価基準として適切な指標の提案

上述の断絶を解消し、実用的な自動受信システムを構築するためには、評価指標の標準化が急務である。以下に、本リサーチに基づいた推奨指標を詳細に定義する。

### 5.1 【最重要】帯域固定SNR（Fixed-Bandwidth SNR: SNRref​）

機械学習モデルの入力サンプリングレートや前処理（STFT等）に依存せず、物理的な信号強度を定義するために、全てのSNRを **2500Hz 帯域幅** に正規化して表記することを提案する。

#### 5.1.1 定義

$$ SNR_{ref} (dB) = SNR_{input} (dB) + 10 \log_{10} \left( \frac{BW_{input}}{2500} \right) $$

ここで、

- $SNR_{input}$ : モデルに入力されるデジタル波形データ上で計算された生のSNR（信号電力/全ノイズ電力）。
- $BW_{input}$ : 入力データの有効帯域幅（通常は  $F_s / 2$ ）。
- $2500$ : 参照帯域幅（Hz）。

#### 5.1.2 算出ロジックの実装（Python例）

データセット生成時や評価時に、以下のロジックでノイズを付加する。

Python

```
import numpy as np

def add_noise_normalized(signal, target_snr_2500hz, fs):
    """
    2500Hz帯域固定SNRに基づいてノイズを付加する関数
    
    Args:
        signal (np.array): 元のCW信号
        target_snr_2500hz (float): 目標とするSNR (dB, @2500Hz)
        fs (int): サンプリング周波数 (Hz)
        
    Returns:
        noisy_signal (np.array): ノイズ付加後の信号
    """
    # 1. 信号電力の計算
    signal_power = np.mean(signal ** 2)
    
    # 2. 現在のシステム帯域幅（ナイキスト周波数）
    current_bandwidth = fs / 2.0
    reference_bandwidth = 2500.0
    
    # 3. 帯域幅補正係数の計算
    # 2500HzでのSNRを、現在の帯域幅でのSNRに変換する
    # 帯域が広がる分、同じN0ならノイズ電力は増えるため、見かけのSNRは下がる
    correction_factor = 10 * np.log10(current_bandwidth / reference_bandwidth)
    
    # データ生成に必要な「生のSNR」
    required_raw_snr = target_snr_2500hz - correction_factor
    
    # 4. ノイズ電力の計算と付加
    noise_power = signal_power / (10 ** (required_raw_snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    return signal + noise

```

**適用の効果**:
この  $SNR_{ref}$  を用いることで、例えばサンプリングレートが 8kHz であろうと 48kHz であろうと、「SNR -10dB」という評価結果は常に**「SSB受信機で聞いた場合のノイズ感」**と一致することになる。

### 5.2 精度評価指標：CERとWER

モールス信号は可変長のシーケンスデータであるため、単純な「正解率（Accuracy）」は不適切である。

#### 5.2.1 文字誤り率（CER: Character Error Rate）

最も推奨される指標である <sup>[[2]](https://github.com/MaorAssayag/morse-deep-learning-detect-and-decode)</sup>。

$$ CER = \frac{S + D + I}{N} \times 100 (\%) $$

- $S$  (Substitutions): 誤置換（例: 'A' を 'B' と誤認）
- $D$  (Deletions): 脱落（例: 文字を見逃す）
- $I$  (Insertions): 挿入（例: ノイズを文字として誤認）
- $N$ : 正解ラベルの総文字数
- 計算には **レーベンシュタイン距離（編集距離）** を用いる。

**なぜCERか**: モールス信号では、1つの「短点」を見逃すだけで、符号の意味が完全に変わる（例: `A` (.-) の短点を落とすと `T` (-) になる）。CERはこの微細なエラーを厳密に測定できる。

#### 5.2.2 単語誤り率（WER: Word Error Rate）

自然言語処理で一般的だが、モールス復号においては補助的な指標とする。ランダムなコールサインや暗号文の受信（和文モールス等）では、1文字の間違いが単語全体の意味を失わせるため、WERは非常に厳しくなる傾向がある。意味のある平文（Plain text）の復号性能を測る場合に有用である。

### 5.3 頑健性（Robustness）の評価項目

SNRだけでなく、以下の項目に対する耐性を評価軸に加えるべきである。

1. **フェージング耐性（QSB）**:

  - レイリー分布（Rayleigh Fading）に基づく信号強度の動的変動をシミュレートする。
  - 特に、信号強度がノイズフロア以下に落ち込む「ディップ」からの回復速度を評価する。
2. **周波数変動耐性（Drift）**:

  - 信号周波数が時間とともに  $\pm 20$  Hz 程度変動する状況下での追従性。
  - CNNベースのモデル（MorseNet等）はこれに強いが、固定フィルタを使用するモデルは弱点となる <sup>[[1]](https://ieeexplore.ieee.org/iel7/6287639/8948470/09183940.pdf)</sup>。
3. **速度変動耐性**:

  - 手打ちモールス（Hand-sent code）特有の速度の揺らぎ（例えば、長点が規定の3倍より長くなる、文字間隔が不均一になる等）に対する許容度。

## 6. 人間と機械の比較分析

本リサーチで得られたデータを統合し、人間（熟練オペレーター）と最新の機械学習モデルの性能を比較する。全てのSNR値は **2500Hz 参照帯域** に正規化されている。

**表2: 人間 vs 機械学習モデル 受信限界SNR比較 (正規化済)**

| 受信主体 | 条件・モデル | 受信限界 SNR (SNRref​) | 情報源・根拠 |
| --- | --- | --- | --- |
| 人間 (神業級) | ZROテスト (数字列) | -18 dB | W2RS |
| 人間 (熟練級) | 一般的なDX通信 | -10 dB 〜 -12 dB | K0NR / 経験値 |
| 人間 (ITU基準) | 商業通信品質 (50%コピー) | -6 dB | ITU-R F.339 |
| MorseNet | 深層学習 (CNN+BiLSTM) | 約 -7 dB (推定) | Li et al. * |
| LSTM-CTC | 深層学習 (RNN) | 約 -5 dB | Assayag * |
| 従来型デコーダ | Goertzel / PLL | +3 dB 〜 +6 dB | 一般的なDSP特性 |
| FT8 (デジタル) | 最新の微弱信号モード | -21 dB | WSJT-X [8] [9] [10] [11] [12] |

**注: MLモデルの数値は、論文中のサンプリングレートやスペクトログラム仕様から2500Hz帯域に換算推定した値。*

### 6.1 考察：人間と機械の「10dBの壁」

表2が示す通り、最新の深層学習モデル（MorseNet等）は、従来のDSPデコーダを大きく凌駕し、ITUが定める「人間が実用的に通信できるレベル（-6 dB）」に肉薄している。これは画期的な成果である。

しかし、人間が極限の集中力で発揮する「-18 dB」の世界には、まだ **10 dB 以上の差** が開いている。この差は、エネルギー的な検出能力の差ではなく、**「認知的補完能力」**の差であると考えられる。

- 人間は、断片的な音からリズムを予測し、言語的な確率（次に来る文字の予測）を瞬時に統合している。
- 機械学習モデルもLSTM等で文脈学習を行っているが、まだ人間の脳の柔軟性（特に未知のノイズパターンやフェージングに対する適応力）には及んでいない。

## 7. 結論と提言

本報告書のリサーチにより、モールス信号の対ノイズ性能評価において、物理層（帯域幅）と情報層（誤り率）を統合した標準化の必要性が明らかになった。

### 7.1 結論

1. **評価指標の不統一**: 機械学習分野でのSNR定義の曖昧さが、実用化への障壁となっている。
2. **人間の優位性**: 帯域幅を正規化した比較において、熟練した人間は依然としてAIモデルに対し約10dBの優位性（マージン）を持っている。
3. **モデルの進化**: CNNとLSTMを組み合わせたモデル（MorseNet等）は、従来の信号処理手法を超え、人間の平均的な能力に近づきつつある。

### 7.2 今後の研究開発への提言

1. **「帯域固定SNR」の採用**: 全ての機械学習ベースのCW復号研究は、SNRを **2500Hz 帯域幅**（またはITU準拠の帯域）に正規化して報告すべきである。これにより、異なるサンプリングレートのモデル間や、人間との直接比較が可能になる。
2. **評価用データセットの標準化**: 単なるAWGNだけでなく、**レイリーフェージング**、**インパルスノイズ**、**周波数ドリフト**、**手打ちの速度変動**を含んだ「標準ベンチマークデータセット」の構築が必要である。
3. **ハイブリッド評価**: 評価指標として、純粋な音響的な復号精度（CER）と、言語モデルによる補正後の精度（Semantic Error Rate）を区別して測定することで、音響モデルと言語モデルそれぞれの貢献度を可視化すべきである。

これらの基準を導入することで、モールス信号復号という「枯れた技術」へのAI応用は、新たなフェーズへ進むことができるだろう。最終的な目標は、-20dBのノイズの海から信号を救い上げる、人間の「黄金の耳」をアルゴリズムで再現することにある。

### 引用文献ID一覧

<sup>[[1]](https://ieeexplore.ieee.org/iel7/6287639/8948470/09183940.pdf)</sup>

## 出典

- [**ieeexplore.ieee.org** MorseNet: A Unified Neural Network for Morse Detection and Recognition in Spectrogram - IEEE Xplore 新しいウィンドウで開く](https://ieeexplore.ieee.org/iel7/6287639/8948470/09183940.pdf)
- [**github.com** MaorAssayag/morse-deep-learning-detect-and-decode: Morse Code Decoder & Detector with Deep Learning - GitHub 新しいウィンドウで開く](https://github.com/MaorAssayag/morse-deep-learning-detect-and-decode)
- [**wavewalkerdsp.com** Calculate Signal to Noise Ratio (SNR) in Python Simulation - Wave Walker DSP 新しいウィンドウで開く](https://www.wavewalkerdsp.com/2024/07/01/calculate-signal-to-noise-ratio-snr-in-simulation/)
- [**mathworks.com** How to calculate the SNR within a given bandwidth. - MATLAB Answers - MathWorks 新しいウィンドウで開く](https://www.mathworks.com/matlabcentral/answers/1764940-how-to-calculate-the-snr-within-a-given-bandwidth)
- [**ham.stackexchange.com** Calculating the signal-to-noise ratio for CW (Morse Code) signals? 新しいウィンドウで開く](https://ham.stackexchange.com/questions/15886/calculating-the-signal-to-noise-ratio-for-cw-morse-code-signals)
- [**kf6hi.net** SNR - KF6HI Amateur Radio 新しいウィンドウで開く](https://kf6hi.net/radio/SNR.html)
- [**pa3fwm.nl** Signal/noise ratio of digital amateur modes 新しいウィンドウで開く](https://pa3fwm.nl/technotes/tn09b.html)
- [**k0nr.com** Weak-Signal Performance of Common Modulation Formats - The KØNR Radio Site 新しいウィンドウで開く](https://www.k0nr.com/wordpress/2025/03/weak-signal-performance/)
- [**g1ogy.com** The Weak-Signal Capability of the Human Ear - G1OGY.com 新しいウィンドウで開く](http://www.g1ogy.com/www.n1bug.net/tech/w2rs/The%20Human%20Ear.pdf)
- [**sm5bsz.com** Receiving Weak CW Signals - SM 5 BSZ 新しいウィンドウで開く](https://www.sm5bsz.com/weakcom.htm)
- [**qsl.net** Signal/noise ratio of digital amateur modes - QSL.net 新しいウィンドウで開く](https://www.qsl.net/on7dy/Documentation/Signal%20noise%20ratio%20of%20digital%20amateur%20modes.pdf)
- [**itu.int** RECOMMENDATION ITU-R F.339-8 - Bandwidths, signal-to-noise ratios and fading allowances in HF fixed and land mobile radiocommuni 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-8-201302-I!!PDF-E.pdf)
- [**la3za.blogspot.com** Studies on Morse code recognition - LA3ZA Radio and Electronics 新しいウィンドウで開く](https://la3za.blogspot.com/2013/10/studies-on-morse-code-recognition.html)
- [**researchgate.net** MorseNet: A Unified Neural Network for Morse Detection and Recognition in Spectrogram 新しいウィンドウで開く](https://www.researchgate.net/publication/344064119_MorseNet_A_Unified_Neural_Network_for_Morse_Detection_and_Recognition_in_Spectrogram)
- [**souryadey.github.io** Morse Code Datasets for Machine Learning - Sourya Dey 新しいウィンドウで開く](https://souryadey.github.io/research/material/SouryaDey_ICCCNT2018_Presentation.pdf)
- [**github.com** Generate Morse code datasets for training artificial neural networks - GitHub 新しいウィンドウで開く](https://github.com/souryadey/morse-dataset)
- [**reversebeacon.blogspot.com** Understanding Signal-to-Noise Ratio (SNR) - Reverse Beacon 新しいウィンドウで開く](http://reversebeacon.blogspot.com/2014/03/understanding-signal-to-noise-ratio-snr.html)
- [**sto.nato.int** Human-Like Morse Code Decoding Using Machine Learning 新しいウィンドウで開く](https://www.sto.nato.int/document/human-like-morse-code-decoding-using-machine-learning/)
- [**itu.int** F.339-6 - Bandwidths, signal-to-noise ratios and fading allowances in complete systems - ITU 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-6-198607-S!!PDF-E.pdf)
- [**itu.int** RECOMMENDATION ITU-R F.339-7* - Bandwidths, signal-to-noise ratios and fading allowances in complete systems 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-7-200602-S!!MSW-E.doc)
- [**minghsiehece.usc.edu** Best Paper Awards - USC Viterbi - Ming Hsieh Department of Electrical and Computer Engineering - University of Southern California 新しいウィンドウで開く](https://minghsiehece.usc.edu/research/best-paper-awards/)
- [**search.itu.int** Recommendations of the CCIR (Düsseldorf, 1990): Volume I 新しいウィンドウで開く](https://search.itu.int/history/HistoryDigitalCollectionDocLibrary/4.283.43.en.1001.pdf)
- [**search.itu.int** Recommendations of the CCIR (Düsseldorf, 1990): Volume III 新しいウィンドウで開く](https://search.itu.int/history/HistoryDigitalCollectionDocLibrary/4.283.43.en.1005.pdf)
- [**itu.int** LIST OF ITU-R RECOMMENDATIONS EDITION MAY 2004 新しいウィンドウで開く](https://www.itu.int/dms_pub/itu-r/opb/rec/r-rec-ls-2004-pdf-e.pdf)
- [**search.itu.int** Recommendations and Reports of the CCIR (Geneva, 1982): Volume VII 新しいウィンドウで開く](https://search.itu.int/history/HistoryDigitalCollectionDocLibrary/4.281.43.en.1008.pdf)
- [**docs.wind-watch.org** Human hearing at low frequencies - National Wind Watch 新しいウィンドウで開く](https://docs.wind-watch.org/Pedersen-human-hearing-low-frequencies.pdf)
- [**pmc.ncbi.nlm.nih.gov** Estimates of Human Cochlear Tuning at Low Levels Using Forward and Simultaneous Masking - PMC - NIH 新しいウィンドウで開く](https://pmc.ncbi.nlm.nih.gov/articles/PMC3202745/)
- [**ham.stackexchange.com** What is the most common CW audio filter center frequency? 新しいウィンドウで開く](https://ham.stackexchange.com/questions/15686/what-is-the-most-common-cw-audio-filter-center-frequency)
- [**pmc.ncbi.nlm.nih.gov** Auditory filter shapes derived from forward and simultaneous masking at low frequencies: Implications for human cochlear tuning - NIH 新しいウィンドウで開く](https://pmc.ncbi.nlm.nih.gov/articles/PMC9167757/)
- [**pubs.aip.org** Proceedings of Meetings on Acoustics - AIP Publishing 新しいウィンドウで開く](https://pubs.aip.org/asa/poma/article-pdf/doi/10.1121/1.4799223/18243917/pma.v19.i1.050184_1.online.pdf)
- [**kb6nu.com** AG1LE challenges developers to come up with better Morse code reader 新しいウィンドウで開く](https://www.kb6nu.com/ag1le-challenges-developers-to-come-up-with-better-morse-code-reader/)
- [**eham.net** Morse Learning Machine Challenge Catching On with Hams: - eHam.net 新しいウィンドウで開く](https://www.eham.net/article/33127)
- [**iasj.rdd.edu.iq** Improvement of signal detection based on using machine learning - Engineering and Technology Journal 新しいウィンドウで開く](https://iasj.rdd.edu.iq/journals/uploads/2025/05/07/666ac9ee609fadd43a645dc1d224e5f5.pdf)
- [**mdpi.com** An SNR Estimation Technique Based on Deep Learning - MDPI 新しいウィンドウで開く](https://www.mdpi.com/2079-9292/8/10/1139)
- [**reddit.com** Morse Learning Machine challenge : r/MachineLearning - Reddit 新しいウィンドウで開く](https://www.reddit.com/r/MachineLearning/comments/2fi18j/morse_learning_machine_challenge/)
- [**kaggle.com** Morse Learning Machine - v1 | Kaggle 新しいウィンドウで開く](https://www.kaggle.com/c/morse-challenge)
- [**reddit.com** I built a CW decoder runs in your browser using a deep learning model - Reddit 新しいウィンドウで開く](https://www.reddit.com/r/amateurradio/comments/1n7ecdf/i_built_a_cw_decoder_runs_in_your_browser_using_a/)
- [**youtube.com** Morse Code Decoder & Detector with Deep Learning - YouTube 新しいウィンドウで開く](https://www.youtube.com/watch?v=uDLtp_Y9Fo4)
- [**youtube.com** python snr calculation - YouTube 新しいウィンドウで開く](https://www.youtube.com/watch?v=Cd2AUToMd7Q)
- [**stackoverflow.com** how to calculate signal to noise ratio using python - Stack Overflow 新しいウィンドウで開く](https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python)
- [**github.com** hrtlacek/SNR: Signal to noise ratio in python - GitHub 新しいウィンドウで開く](https://github.com/hrtlacek/SNR)
- [**dsp.stackexchange.com** Calculating the SNR of Audio Signal (Recommended Libraries) 新しいウィンドウで開く](https://dsp.stackexchange.com/questions/49577/calculating-the-snr-of-audio-signal-recommended-libraries)
- [**community.flexradio.com** minimum signal strength for cw - FlexRadio Community 新しいウィンドウで開く](https://community.flexradio.com/discussion/8012667/minimum-signal-strength-for-cw)
- [**kk5jy.net** CW Modem - KK5JY.Net 新しいウィンドウで開く](http://www.kk5jy.net/cw-modem-v1/)
- [**robkalmeijer.nl** Measuring SSB/CW receiver sensitivity 新しいウィンドウで開く](https://www.robkalmeijer.nl/techniek/electronica/radiotechniek/hambladen/qst/1992/10/page30/index.html)
- [**youtube.com** Radio Receiver Signal to Noise Ratio SNR Specification - YouTube 新しいウィンドウで開く](https://www.youtube.com/watch?v=WT8p6G-lN0g)
- [**arrl.org** A Software Defined Radio for the Masses, Part 4 - ARRL 新しいウィンドウで開く](https://www.arrl.org/files/file/Technology/tis/info/pdf/030304qex020.pdf)
- [**arrl.org** Improved Dynamic- RangeTesting - ARRL 新しいウィンドウで開く](http://www.arrl.org/files/file/Technology/tis/info/pdf/020708qex046.pdf)
- [**w8ji.com** Mixing Wide and Narrow Modes - W8JI 新しいウィンドウで開く](https://www.w8ji.com/mixing_wide_and_narrow_modes.htm)
- [**itu.int** RECOMMENDATION ITU-R M.1796-3 - Characteristics of and protection criteria for radars operating in the radiodetermination ser 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/m/R-REC-M.1796-3-202202-I!!PDF-E.pdf)
- [**udel.edu** National Bureau of Standards Technical Note 101 新しいウィンドウで開く](https://udel.edu/~mm/itm/lr1.pdf)
- [**ntia.gov** Defense Spectrum Organization - National Telecommunications and Information Administration 新しいウィンドウで開く](https://www.ntia.gov/files/ntia/publications/jsc-cr-10-004final.pdf)
- [**ma-mimo.ellintech.se** When Normalization is Dangerous | Wireless Future Blog 新しいウィンドウで開く](https://ma-mimo.ellintech.se/2018/04/14/when-normalization-is-dangerous/)
- [**dsprelated.com** Difference between C/N and SNR and understanding - DSPRelated.com 新しいウィンドウで開く](https://www.dsprelated.com/showthread/comp.dsp/157234-1.php)
- [**reddit.com** Why does narrowing the bandwidth of a receiver improve S/N ratio? - Reddit 新しいウィンドウで開く](https://www.reddit.com/r/amateurradio/comments/11br9wh/why_does_narrowing_the_bandwidth_of_a_receiver/)
- [**amateurradio.com** Weak-Signal Performance of Common Modulation Formats - AmateurRadio.com 新しいウィンドウで開く](https://www.amateurradio.com/weak-signal-performance-of-common-modulation-formats/)
- [**everythingrf.com** Minimum Detectable Signal Calculator - everything RF 新しいウィンドウで開く](https://www.everythingrf.com/rf-calculators/minimum-detectable-signal-calculator)
- [**ntrs.nasa.gov** Signal to noise ratio calculation for fiber optics links - NASA Technical Reports Server (NTRS) 新しいウィンドウで開く](https://ntrs.nasa.gov/citations/19800021830)
- [**electronics.stackexchange.com** Calculating Data rate function of bandwidth and SNR - Electronics Stack Exchange 新しいウィンドウで開く](https://electronics.stackexchange.com/questions/178684/calculating-data-rate-function-of-bandwidth-and-snr)
- [**ntrs.nasa.gov** N92-22018 Bandwidth Efficient Coding for Satellite Communications* RESEARCH PURPOSE 新しいウィンドウで開く](https://ntrs.nasa.gov/api/citations/19920012775/downloads/19920012775.pdf)
- [**apps.dtic.mil** Quasi-Real Time Translation of Morse-Coded Signals Using Digital Delay Processing - DTIC 新しいウィンドウで開く](https://apps.dtic.mil/sti/tr/pdf/ADA030083.pdf)
- [**mdpi.com** Morse Code Recognition Based on a Flexible Tactile Sensor with Carbon Nanotube/Polyurethane Sponge Material by the Long Short-Term Memory Model - MDPI 新しいウィンドウで開く](https://www.mdpi.com/2072-666X/15/7/864)
- [**pubmed.ncbi.nlm.nih.gov** Characterizing the Speech Reception Threshold in hearing-impaired listeners in relation to masker type and masker level - PubMed 新しいウィンドウで開く](https://pubmed.ncbi.nlm.nih.gov/24606285/)
- [**pmc.ncbi.nlm.nih.gov** The Just-Noticeable Difference in Speech-to-Noise Ratio - PMC - NIH 新しいウィンドウで開く](https://pmc.ncbi.nlm.nih.gov/articles/PMC4335553/)
- [**dsp.stackexchange.com** The minimal signal to noise ratio (SNR) for people to understand a speech in the noisy background 新しいウィンドウで開く](https://dsp.stackexchange.com/questions/38178/the-minimal-signal-to-noise-ratio-snr-for-people-to-understand-a-speech-in-the)
- [**forum.amsat-dl.org** SNR of the CW beacon - LNB for RX - AMSAT-DL Forum 新しいウィンドウで開く](https://forum.amsat-dl.org/index.php?thread/316-snr-of-the-cw-beacon/)
- [**en.wikipedia.org** X.690 - Wikipedia 新しいウィンドウで開く](https://en.wikipedia.org/wiki/X.690)
- [**learn.microsoft.com** Distinguished Encoding Rules - Win32 apps - Microsoft Learn 新しいウィンドウで開く](https://learn.microsoft.com/en-us/windows/win32/seccertenroll/distinguished-encoding-rules)
- [**oss.com** ASN.1 Encoding Rules Overview and Examples - OSS Nokalva 新しいウィンドウで開く](https://www.oss.com/asn1/resources/asn1-made-simple/encoding-rules.html)
- [**itu.int** Word 2007 - ITU 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-8-201302-I!!MSW-E.docx)
- [**researchgate.net** Bit error rate vs signal‐to‐noise ratio performances of the coded and... - ResearchGate 新しいウィンドウで開く](https://www.researchgate.net/figure/Bit-error-rate-vs-signal-to-noise-ratio-performances-of-the-coded-and-uncoded-systems_fig2_340623046)
- [**en.wikipedia.org** Bit error rate - Wikipedia 新しいウィンドウで開く](https://en.wikipedia.org/wiki/Bit_error_rate)
- [**ntrs.nasa.gov** Bit Error Rate and Frame Error Rate Data Processing for Space Communications and Navigation-Related Communication System Analysis Tools 新しいウィンドウで開く](https://ntrs.nasa.gov/api/citations/20190026442/downloads/20190026442.pdf)
- [**reddit.com** Struggling to understand the difference between plotted curves on an SNR vs BER relating to coding theory : r/DSP - Reddit 新しいウィンドウで開く](https://www.reddit.com/r/DSP/comments/rb07mi/struggling_to_understand_the_difference_between/)
- [**youtube.com** Bit Error Rate (BER) and Signal to Noise Ratio (SNR) - YouTube 新しいウィンドウで開く](https://www.youtube.com/watch?v=UOLRP52oOPI)
- [**mne.discourse.group** SNR estimate for real MEG data - Mailing List Archive (read-only) - MNE Forum 新しいウィンドウで開く](https://mne.discourse.group/t/snr-estimate-for-real-meg-data/1640)
- [**mspass.org** Signal to Noise Ratio Estimation — MsPASS 0.0.1 documentation 新しいウィンドウで開く](https://www.mspass.org/user_manual/signal_to_noise.html)
- [**youtube.com** Estimating Signal-to-Noise Ratio (SNR) in Hyperspectral Imaging - YouTube 新しいウィンドウで開く](https://www.youtube.com/watch?v=j2NQyQl6KrA)
- [**horiba.com** How to Calculate Signal to Noise Ratio - HORIBA 新しいウィンドウで開く](https://www.horiba.com/usa/scientific/technologies/fluorescence-spectroscopy/how-to-calculate-signal-to-noise-ratio/)
- [**apps.dtic.mil** Machine Recognition of Hand-Sent Morse Code Using the PDP-12 Computer - DTIC 新しいウィンドウで開く](https://apps.dtic.mil/sti/tr/pdf/AD0786492.pdf)
- [**atlantis-press.com** Automatic Morse Code Recognition Under Low SNR Xianyu Wanga, Qi Zhaob, Cheng Mac, * and Jianping Xiongd - Atlantis Press 新しいウィンドウで開く](https://www.atlantis-press.com/article/25893679.pdf)
- [**arxiv.org** Morse Code-Enabled Speech Recognition for Individuals with Visual and Hearing Impairments - arXiv 新しいウィンドウで開く](https://arxiv.org/html/2407.14525v1)
- [**itu.int** RECOMMENDATION ITU-R F.339-7* - Bandwidths, signal-to-noise ratios and fading allowances in complete systems 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-7-200602-S!!PDF-E.pdf)
- [**itu.int** F.339-6 - Bandwidths, signal-to-noise ratios and fading allowances in complete systems - ITU 新しいウィンドウで開く](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.339-6-198607-S!!MSW-E.doc)
- [**itu.int** F.339 : Bandwidths, signal-to-noise ratios and fading allowances in HF fixed and land mobile radiocommunication systems - ITU 新しいウィンドウで開く](https://www.itu.int/rec/R-REC-F.339/en)
- [**scholar.google.com.mx** ‪Sourya Dey‬ - ‪Google Scholar‬ 新しいウィンドウで開く](https://scholar.google.com.mx/citations?user=llcYR9MAAAAJ&hl=fil)
- [**researchgate.net** Quantitative comparison results on four datasets. The input of... - ResearchGate 新しいウィンドウで開く](https://www.researchgate.net/figure/Quantitative-comparison-results-on-four-datasets-The-input-of-recognition-is-the-output_tbl2_344064119)
- [**semanticscholar.org** [PDF] DeepMorse: A Deep Convolutional Learning Method for Blind 新しいウィンドウで開く](https://www.semanticscholar.org/paper/DeepMorse%3A-A-Deep-Convolutional-Learning-Method-for-Yuan-Sun/d02b0b845636e407c9a78851a3766d96f16b536b)
- [**k0nr.com** VHF/UHF Archives - The KØNR Radio Site 新しいウィンドウで開く](https://www.k0nr.com/wordpress/category/vhf/)
- [**veronfriesemeren.nl** 25 Jan-Feb SARC Communicator - VERON Friese Meren 新しいウィンドウで開く](https://veronfriesemeren.nl/wordpress/wp-content/uploads/2025/05/May-Jun-SARC-CommunicatorC.pdf)
- [**mdpi.com** Contactless Heart and Respiration Rates Estimation and Classification of Driver Physiological States Using CW Radar and Temporal Neural Networks - MDPI 新しいウィンドウで開く](https://www.mdpi.com/1424-8220/23/23/9457)
- [**dsp.stackexchange.com** Signal to noise ratio (SNR) of a CW signal 新しいウィンドウで開く](https://dsp.stackexchange.com/questions/75956/signal-to-noise-ratio-snr-of-a-cw-signal)
- [**pubmed.ncbi.nlm.nih.gov** Reception of Morse code through motional, vibrotactile, and auditory stimulation - PubMed 新しいウィンドウで開く](https://pubmed.ncbi.nlm.nih.gov/9360474/)
- [**digitalcommons.usf.edu** Designing the Haptic Interface for Morse Code - Digital Commons @ USF - University of South Florida 新しいウィンドウで開く](https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=7797&context=etd)
- [**arxiv.org** Utilizing Machine Learning for Signal Classification and Noise Reduction in Amateur Radio 新しいウィンドウで開く](https://arxiv.org/html/2402.17771v1)
- [**scholarworks.calstate.edu** an examination of the use of matched filters in an automatic Morse code receiver | ScholarWorks 新しいウィンドウで開く](https://scholarworks.calstate.edu/concern/theses/9p290f013?locale=it)
- [**researchgate.net** (PDF) Reception of Morse code through motional, vibrotactile, and auditory stimulation 新しいウィンドウで開く](https://www.researchgate.net/publication/225794908_Reception_of_Morse_code_through_motional_vibrotactile_and_auditory_stimulation)
- [**ijarst.in** DECRYPTION OF MORSE CODE FROM VOICE USING A CONVOLUTIONAL NEURAL NETWORK - IJARST 新しいウィンドウで開く](https://www.ijarst.in/public/uploads/paper/258911689418215.pdf)
- [**patents.google.com** CN106650605A - Morse signal automatic detection decoding method based on machine learning - Google Patents 新しいウィンドウで開く](https://patents.google.com/patent/CN106650605A/en)
- [**rohde-schwarz.com** Understanding basic spectrum analyzer operation | Rohde & Schwarz 新しいウィンドウで開く](https://www.rohde-schwarz.com/us/products/test-and-measurement/essentials-test-equipment/spectrum-analyzers/understanding-basic-spectrum-analyzer-operation_256005.html)
- [**en.wikipedia.org** Signal-to-noise ratio - Wikipedia 新しいウィンドウで開く](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)
- [**qsl.net** RADIO NERD AND CW - OTHER MODES COMPARISON! - QSL.net 新しいウィンドウで開く](https://www.qsl.net/pa2ohh/24nerd.htm)
- [**eevblog.com** How fast can Morse code be sent - EEVblog 新しいウィンドウで開く](https://www.eevblog.com/forum/rf-microwave/how-fast-can-morse-code-be-sent/)
- [**kb6nu.com** CW Geek's Guide to Having Fun With Morse Code: Getting on the Air – Tuning In 新しいウィンドウで開く](https://www.kb6nu.com/cw-geeks-guide-to-having-fun-with-morse-code-getting-on-the-air-tuning-in/)
- [**en.wikipedia.org** Minimum detectable signal - Wikipedia 新しいウィンドウで開く](https://en.wikipedia.org/wiki/Minimum_detectable_signal)
- [**radartutorial.eu** Minimum Detectable Signal - MDS - Radartutorial.eu 新しいウィンドウで開く](https://www.radartutorial.eu/09.receivers/rx51.en.html)
- [**ittc.ku.edu** Minimum Detectable Signal 新しいウィンドウで開く](http://www.ittc.ku.edu/~jstiles/622/handouts/Minimum%20Detectable%20Signal.pdf)
- [**rfcafe.com** Electronic Warfare and Radar Systems Engineering Handbook - Receiver Sensitivity / Noise 新しいウィンドウで開く](https://www.rfcafe.com/references/electrical/ew-radar-handbook/receiver-sensitivity-noise.htm)
- [**researchgate.net** (PDF) A Robust Real-Time Automatic Recognition Prototype for Maritime Optical Morse-Based Communication Employing Modified Clustering Algorithm - ResearchGate 新しいウィンドウで開く](https://www.researchgate.net/publication/339276119_A_Robust_Real-Time_Automatic_Recognition_Prototype_for_Maritime_Optical_Morse-Based_Communication_Employing_Modified_Clustering_Algorithm)
- [**researchgate.net** A Deep Convolutional Network for Multitype Signal Detection and Classification in Spectrogram - ResearchGate 新しいウィンドウで開く](https://www.researchgate.net/publication/344480452_A_Deep_Convolutional_Network_for_Multitype_Signal_Detection_and_Classification_in_Spectrogram)
- [**theses.lib.polyu.edu.hk** Copyright Undertaking - PolyU Electronic Theses 新しいウィンドウで開く](https://theses.lib.polyu.edu.hk/bitstream/200/12797/3/7248.pdf)
- [**uhra.herts.ac.uk** Reducing Errors in Optical Data Transmission Using Trainable Machine Learning Methods 新しいウィンドウで開く](https://uhra.herts.ac.uk/id/eprint/16735/1/14077110%20BINJUMAH%20Weam%20Final%20Version%20of%20PhD%20Submission.pdf)
- [**hackaday.com** Machine Learning System Uses Images To Teach Itself Morse Code | Hackaday 新しいウィンドウで開く](https://hackaday.com/2020/01/27/machine-learning-system-uses-images-to-teach-itself-morse-code/)
- [**panoradio-sdr.de** Automatic Identification of 160 Shortwave RF Signals with Deep Learning - Panoradio SDR 新しいウィンドウで開く](https://panoradio-sdr.de/automatic-identification-of-160-shortwave-rf-signals-with-deep-learning/)
- [**pmc.ncbi.nlm.nih.gov** Multi-Signal Detection Framework: A Deep Learning Based Carrier Frequency and Bandwidth Estimation - PMC - NIH 新しいウィンドウで開く](https://pmc.ncbi.nlm.nih.gov/articles/PMC9147498/)
- [**pmc.ncbi.nlm.nih.gov** Verification of Estimated Output Signal-to-Noise Ratios From a Phase Inversion Technique Using a Simulated Hearing Aid - PMC - NIH 新しいウィンドウで開く](https://pmc.ncbi.nlm.nih.gov/articles/PMC10166192/)
- [**semanticscholar.org** [PDF] Morse Code Datasets for Machine Learning - Semantic Scholar 新しいウィンドウで開く](https://www.semanticscholar.org/paper/Morse-Code-Datasets-for-Machine-Learning-Dey-Chugg/891a5cae04b884830e7c712f3335532d63283509)
- [**researchgate.net** (PDF) Morse Code Datasets for Machine Learning - ResearchGate 新しいウィンドウで開く](https://www.researchgate.net/publication/328761726_Morse_Code_Datasets_for_Machine_Learning)
- [**souryadey.github.io** EXPLORING COMPLEXITY REDUCTION FOR LEARNING IN DEEP NEURAL NETWORKS by Sourya Dey PhD Dissertation Proposal UNIVERSITY OF SOUTHE 新しいウィンドウで開く](https://souryadey.github.io/research/material/SouryaDey_PhDDissertationProposal2019.pdf)
- [**g1ogy.com** N1BUG Web: The Weak-Signal Capability of the Human Ear - G1OGY.com 新しいウィンドウで開く](http://www.g1ogy.com/www.n1bug.net/tech/w2rs/humanear.html)
- [**pmc.ncbi.nlm.nih.gov** Characteristics of Real-World Signal-to-noise Ratios and Speech Listening Situations of Older Adults with Mild-to-Moderate Hearing Loss - PMC - NIH 新しいウィンドウで開く](https://pmc.ncbi.nlm.nih.gov/articles/PMC5824438/)
- [**kaggle.com** Morse Learning Machine Challenge - v2 - Kaggle 新しいウィンドウで開く](https://www.kaggle.com/c/morse-learning-machine-challenge-v2/data)
- [**kaggle.com** Morse Learning Machine - v1 - Kaggle 新しいウィンドウで開く](https://www.kaggle.com/c/morse-challenge/data)
- [**cobaltfolly.wordpress.com** Morse code dataset for Artificial Neural Networks - cobaltfolly - WordPress.com 新しいウィンドウで開く](https://cobaltfolly.wordpress.com/2017/10/15/morse-code-dataset-for-artificial-neural-networks/)
- [**arxiv.org** Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models - arXiv 新しいウィンドウで開く](https://arxiv.org/html/2506.18251v1)
- [**reddit.com** Real-time deep learning Morse decoder experiment : r/amateurradio - Reddit 新しいウィンドウで開く](https://www.reddit.com/r/amateurradio/comments/fzyr7g/realtime_deep_learning_morse_decoder_experiment/)
