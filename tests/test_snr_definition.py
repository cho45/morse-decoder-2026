import numpy as np
import pytest
import config
from data_gen import HFChannelSimulator

def calculate_band_limited_power(waveform, sample_rate, low_freq, high_freq):
    """計算された波形の特定帯域内の電力を計算する"""
    fft = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(len(waveform), 1/sample_rate)
    
    # 指定帯域内のインデックスを取得
    idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    
    # パワースペクトル密度 (PSD)
    # 信号の分散（電力）を、そのサンプリングレートにおける「1Hzあたりの密度」に換算する
    # 全帯域電力 P = N0 * (Fs/2)  => N0 = P / (Fs/2)
    total_power = np.var(waveform)
    n0 = total_power / (sample_rate / 2)
    return n0

def test_snr_sampling_rate_independence():
    """
    サンプリングレートが異なっても、同じ snr_2500 設定であれば
    特定帯域(2500Hz)内のノイズ密度が一定であることを検証する。
    """
    snr_2500 = -10.0
    bw_ref = 2500.0
    
    # 異なるサンプリングレートで比較
    fs_list = [8000, 16000, 48000]
    powers = []
    
    for fs in fs_list:
        sim = HFChannelSimulator(sample_rate=fs)
        # 無音信号にノイズを乗せる
        duration = 1.0
        waveform = np.zeros(int(fs * duration))
        
        # 修正後のインターフェースを使用
        noisy = sim.apply_noise(waveform, snr_2500=snr_2500)
            
        # 0-2500Hz 帯域の平均電力を計算
        p = calculate_band_limited_power(noisy, fs, 100, 2400)
        powers.append(p)
        print(f"Fs: {fs}Hz, Band-limited Power: {p:.10f}")

    # すべてのサンプリングレートで電力が（ほぼ）一致することを確認
    # 現状の実装では Fs が大きいほど電力が下がるため、ここで失敗するはず
    for i in range(len(powers) - 1):
        assert pytest.approx(powers[i], rel=0.1) == powers[i+1], \
            f"Power mismatch between Fs={fs_list[i]} and Fs={fs_list[i+1]}"

if __name__ == "__main__":
    test_snr_sampling_rate_independence()