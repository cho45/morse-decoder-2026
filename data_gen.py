"""
CW Data Generator module (Refactored).
Responsible for synthesizing Morse code signals with human keying artifacts
and HF channel simulations (noise, fading, QRM).
"""

import torch
import numpy as np
import scipy.signal
import scipy.io.wavfile
import random
import string
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
import config
from config import (
    SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE,
    INTERNAL_ID_INTRA_GAP, INTERNAL_ID_INTER_CHAR
)

# Morse Code Definition
MORSE_DICT = {
	"A":".-",
	"B":"-...",
	"C":"-.-.",
	"D":"-..",
	"E":".",
	"F":"..-.",
	"G":"--.",
	"H":"....",
	"I":"..",
	"J":".---",
	"K":"-.-",
	"L":".-..",
	"M":"--",
	"N":"-.",
	"O":"---",
	"P":".--.",
	"Q":"--.-",
	"R":".-.",
	"S":"...",
	"T":"-",
	"U":"..-",
	"V":"...-",
	"W":".--",
	"X":"-..-",
	"Y":"-.--",
	"Z":"--..",
	"0":"-----",
	"1":".----",
	"2":"..---",
	"3":"...--",
	"4":"....-",
	"5":".....",
	"6":"-....",
	"7":"--...",
	"8":"---..",
	"9":"----.",
	".":".-.-.-",
	",":"--..--",
	"?":"..--..",
	"'":".----.",
	"!":"-.-.--",
	"/":"-..-.",
	"(":"-.--.",
	")":"-.--.-",
	"&":".-...", # AS
	":":"---...",
	";":"-.-.-.",
	"=":"-...-", # BT (new paragraph)
	"+":".-.-.",  # AR (end of message)
	"-":"-....-",
	"_":"..--.-",
	"\"":".-..-.",
	"$":"...-..-",
	"@":".--.-.",
	"<AA>" : ".-.-", # AA (new line)
    "<KA>" : "-.-.-", # CT/KA (attention)
    "<SK>" : "...-.-", # VA/SK (end of transmission)
    '<VE>': '...-.',
    '<HH>': '........',
    '<NJ>': '-..---',
    '<DDD>': '-..-..-..',
    '<SOS>': '...---...',
}


class MorseEncoder:
    """テキストとモールスコード/トークンの相互変換（ユーティリティ）"""
    
    def text_to_morse_code(self, text: str) -> str:
        """テキストをモールスコードに変換
        
        重要: Prosigns（<SK>, <KA> など）を正しく扱う
        """
        tokens = self.text_to_tokens(text)
        morse_codes = []
        for token in tokens:
            if token == ' ':
                continue  # スペースはスキップ（元の実装と同じ挙動）
            else:
                morse_codes.append(MORSE_DICT.get(token, ""))
        return " ".join(morse_codes)
    
    def text_to_tokens(self, text: str) -> List[str]:
        """テキストをトークンリストに変換
        
        重要: Prosigns（<SK>, <KA> など）を正しく扱う
        """
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == '<':
                end = text.find('>', i)
                if end != -1:
                    token = text[i:end+1]
                    # Check for Long Gap token
                    if token.startswith('<GAP:') and token.endswith('>'):
                        tokens.append(token)
                        i = end + 1
                        continue
                    # Check for Prosigns
                    if token in MORSE_DICT:
                        tokens.append(token)
                        i = end + 1
                        continue
            
            # Check for multi-char tokens like CQ, DE if they are treated as single tokens in config
            found = False
            for token in config.PROSIGNS:
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    found = True
                    break
            if found: continue

            # Space is now a valid token in config.CHARS
            tokens.append(text[i])
            i += 1
        return tokens
    
    def count_units_in_tokens(self, tokens: List[str], weight: float = 1.0) -> int:
        """トークンリストの総ユニット数を計算"""
        total_units = 0
        for i, token in enumerate(tokens):
            if token == ' ':
                total_units += 7
                continue
            code = MORSE_DICT.get(token, "")
            for k, symbol in enumerate(code):
                if symbol == '.': total_units += 1 * weight
                elif symbol == '-': total_units += 3 * weight
                # Intra-char space
                if k < len(code) - 1:
                    total_units += 1
            
            # Add inter-char space after token (only if not last token)
            if i < len(tokens) - 1 and tokens[i+1] != ' ':
                total_units += 3
        return total_units

    def tokens_to_text(self, tokens: List[str]) -> str:
        """トークンリストをテキストに結合（逆変換）"""
        text = "".join(tokens)
        return text


class TimingGenerator:
    """トークンからタイミングシーケンスの生成と推定"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE,
                 encoder: MorseEncoder = None):
        self.sample_rate = sample_rate
        self.encoder = encoder or MorseEncoder()
    
    def _parse_gap_token(self, token: str) -> Tuple[bool, Optional[float]]:
        """
        特殊トークン <GAP:duration> を解析
        
        Args:
            token: トークン文字列
        
        Returns:
            (is_gap, duration): is_gap は True の場合、duration は Long Gap の持続時間（秒）
        """
        if token.startswith('<GAP:') and token.endswith('>'):
            try:
                duration = float(token[5:-1])  # <GAP:0.5> -> 0.5
                return True, duration
            except ValueError:
                return False, None
        return False, None
    
    def _morse_code_to_timing(self, morse_code: str, dot_samples: int,
                           weight: float, jitter: float) -> List[Tuple[int, int]]:
        """モールスコードをタイミング（サンプル数）に変換（内部使用）"""
        timing = []
        for k, symbol in enumerate(morse_code):
            if symbol == '.':
                num_samples = int(round(dot_samples * weight))
                timing.append((SIG_ID_DIT, num_samples))  # 短点
            elif symbol == '-':
                num_samples = int(round(dot_samples * 3 * weight))
                timing.append((SIG_ID_DAH, num_samples))  # 長点
            elif symbol == ' ':
                continue  # Inter-char space は外で処理
            
            if jitter > 0:
                # 整数サンプル数に対して jitter を適用
                jittered_samples = int(round(timing[-1][1] * (1 + random.uniform(-jitter, jitter))))
                timing[-1] = (timing[-1][0], jittered_samples)
            
            if k < len(morse_code) - 1 and morse_code[k+1] != ' ':
                # 文字内の要素間ギャップ (1 unit)
                timing.append((INTERNAL_ID_INTRA_GAP, dot_samples))
        
        return timing
    
    def generate_timing(self, text: str, wpm: int = 20,
                        farnsworth_wpm: int = None, jitter: float = 0.0,
                        weight: float = 1.0, pre_silence: float = None,
                        max_duration: float = config.TRAIN_DURATION) -> List[Tuple[int, int]]:
        """
        テキストからタイミングシーケンスを生成（サンプル数ベース）
        Classes: 1: Dit, 2: Dah, 3: Intra-char space, 4: Inter-char space, 5: Inter-word space
        """
        if farnsworth_wpm is None:
            farnsworth_wpm = wpm
        
        # Calculate samples per unit
        dot_samples = int(round((1.2 / wpm) * self.sample_rate))
        char_space_samples = int(round((3 * 1.2 / farnsworth_wpm) * self.sample_rate))
        word_space_samples = int(round((7 * 1.2 / farnsworth_wpm) * self.sample_rate))
        
        target_total_samples = None
        if max_duration is not None:
            target_total_samples = int(round(max_duration * self.sample_rate))
        
        timing = []
        current_samples = 0
        
        # Calculate pre-silence samples
        if pre_silence is not None:
            pre_samples = int(round(pre_silence * self.sample_rate))
            # 既に max_duration を超えていないかチェック
            if target_total_samples is not None and pre_samples > target_total_samples:
                pre_samples = target_total_samples
            
            if pre_samples > 0:
                timing.append((SIG_ID_BLANK, pre_samples))
                current_samples += pre_samples

        words_raw = text.split(' ')
        
        for i, word_raw in enumerate(words_raw):
            if target_total_samples is not None and current_samples >= target_total_samples:
                break
                
            if not word_raw and i < len(words_raw) - 1:
                continue
            tokens = self.encoder.text_to_tokens(word_raw)
            
            word_completed = False
            for j, char in enumerate(tokens):
                if target_total_samples is not None and current_samples >= target_total_samples:
                    break
                    
                # Check for special GAP token
                is_gap, gap_duration = self._parse_gap_token(char)
                if is_gap:
                    gap_samples = int(round(gap_duration * self.sample_rate))
                    if target_total_samples is not None:
                        num_samples = min(gap_samples, target_total_samples - current_samples)
                    else:
                        num_samples = gap_samples
                        
                    if num_samples > 0:
                        timing.append((SIG_ID_BLANK, num_samples))
                        current_samples += num_samples
                    continue
                
                # Regular token
                code = MORSE_DICT.get(char, "")
                if not code:
                    continue
                
                # Generate timing for this token (all ints)
                token_timing = self._morse_code_to_timing(code, dot_samples, weight, jitter)
                
                # トークン全体と、それに続く必須スペースが target_total_samples に収まるかチェック
                token_samples = sum(t[1] for t in token_timing)
                
                needed_space = 0
                if j < len(tokens) - 1:
                    needed_space = char_space_samples
                elif i < len(words_raw) - 1:
                    needed_space = word_space_samples
                
                if target_total_samples is not None and current_samples + token_samples + needed_space > target_total_samples:
                    # これ以上文字（と必須スペース）を追加できないため、終了。
                    break
                
                timing.extend(token_timing)
                current_samples += token_samples
                
                # Add inter-char space after token (only if not last token in word)
                if j < len(tokens) - 1:
                    timing.append((INTERNAL_ID_INTER_CHAR, char_space_samples))
                    current_samples += char_space_samples
            else:
                # Word completed successfully (no break)
                word_completed = True
                # Add inter-word space after word (only if not last word)
                if i < len(words_raw) - 1:
                    timing.append((SIG_ID_WORD_SPACE, word_space_samples))
                    current_samples += word_space_samples
            
            # If word was not completed (inner loop broke), stop everything
            if not word_completed:
                break
        
        # 最後に padding (post_silence) を追加して正確に target_total_samples に合わせる
        if target_total_samples is not None and current_samples < target_total_samples:
            padding = target_total_samples - current_samples
            if padding > 0:
                timing.append((SIG_ID_BLANK, padding))
                current_samples += padding
            
        return timing
    
    def estimate_wpm_for_target_frames(self, tokens: List[str],
                                       target_frames: int = config.TARGET_FRAMES,
                                       min_wpm: int = 10, max_wpm: int = 45) -> int:
        """ターゲットフレーム数に収めるためのWPMを推定"""
        total_units = self.encoder.count_units_in_tokens(tokens)
        
        # GAP トークンの固定時間を合計
        gap_total_sec = 0.0
        for t in tokens:
            is_gap, duration = self._parse_gap_token(t)
            if is_gap:
                gap_total_sec += duration
        
        target_sec = (target_frames * config.HOP_LENGTH / self.sample_rate) - gap_total_sec
        
        # 符号のために使える時間が極端に少ない場合は、安全のために最大WPMを返す
        if target_sec <= 0.1: return max_wpm
        
        needed_wpm = (1.2 * total_units) / target_sec
        
        return int(np.clip(needed_wpm, min_wpm, max_wpm))
    
    def estimate_max_tokens_for_wpm(self, wpm: int,
                                   target_frames: int = config.TARGET_FRAMES) -> int:
        """指定WPMで収まる最大トークン数を推定"""
        target_sec = max(0.5, (target_frames * config.HOP_LENGTH / self.sample_rate) - 0.2)
        total_units = (target_sec * wpm) / 1.2
        
        # Average units per character:
        # PARIS is 50 units for 5 chars + 1 space = 8.33 units/char.
        # We use 13 units/char (less conservative than 15) to increase density while keeping buffer.
        max_tokens = int(total_units / 13)
        return max(1, max_tokens)
    
    def estimate_duration(self, tokens: List[str], wpm: int,
                          farnsworth_wpm: int = None, weight: float = 1.0) -> float:
        """
        トークンの持続時間を推定（秒）
        generate_timing を使用して正確な長さを計算する。
        """
        text = self.encoder.tokens_to_text(tokens)
        timing = self.generate_timing(text, wpm=wpm, farnsworth_wpm=farnsworth_wpm,
                                      weight=weight, max_duration=None)
        total_samples = sum(t[1] for t in timing)
        return total_samples / self.sample_rate


class WaveformGenerator:
    """タイミングシーケンスから音声波形を生成"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def generate_waveform(self, timing: List[Tuple[int, int]],
                          frequency: float = 700.0, waveform_type: str = 'sine',
                          rise_time: float = 0.005, drift_hz: float = 0.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        タイミングシーケンスを音声波形に変換
        """
        total_samples = sum(t[1] for t in timing)
        waveform = np.zeros(total_samples)
        phase = 0.0
        current_sample = 0
        
        for class_id, num_samples in timing:
            if current_sample >= total_samples:
                break
            
            if class_id in [SIG_ID_DIT, SIG_ID_DAH]:
                # 信号生成
                end_sample = min(current_sample + num_samples, total_samples)
                actual_num = end_sample - current_sample
                if actual_num <= 0: break
                
                # ドリフトを考慮した位相計算
                if drift_hz > 0:
                    inst_freq = frequency + drift_hz * np.sin(2 * np.pi * 0.2 * (current_sample + np.arange(actual_num)) / self.sample_rate)
                else:
                    inst_freq = np.full(actual_num, frequency)
                
                d_phase = 2 * np.pi * inst_freq / self.sample_rate
                sig_phase = phase + np.cumsum(d_phase)
                phase = sig_phase[-1] % (2 * np.pi)
                
                sig = np.sin(sig_phase)
                
                # エンベロープ適用
                n_rise = int(rise_time * self.sample_rate)
                if n_rise * 2 > actual_num:
                    n_rise = actual_num // 2
                if n_rise > 0:
                    rise = 0.5 * (1 - np.cos(np.pi * np.arange(n_rise) / n_rise))
                    sig[:n_rise] *= rise
                    sig[-n_rise:] *= rise[::-1]
                
                waveform[current_sample:end_sample] = sig
            
            current_sample += num_samples
            
        return waveform, timing
    
    def _apply_envelope(self, sig: np.ndarray, rise_time: float) -> np.ndarray:
        """エンベロープ（ライズ/フォール）を適用"""
        n_rise = int(rise_time * self.sample_rate)
        if n_rise * 2 > len(sig):
            n_rise = len(sig) // 2
        
        envelope = np.ones(len(sig))
        if n_rise > 0:
            rise = 0.5 * (1 - np.cos(np.pi * np.arange(n_rise) / n_rise))
            envelope[:n_rise] = rise
            envelope[-n_rise:] = rise[::-1]
        
        return sig * envelope
    
    def _generate_tone(self, num_samples: int, frequency: float,
                       phase: float, drift_hz: float) -> Tuple[np.ndarray, float]:
        """トーン波形を生成（位相返りあり）"""
        t = np.arange(num_samples) / self.sample_rate
        
        if drift_hz > 0:
            # Slow sinusoidal drift
            inst_freq = frequency + drift_hz * np.sin(2 * np.pi * 0.2 * t)
        else:
            inst_freq = np.full(num_samples, frequency)
        
        # Update phase based on instantaneous frequency
        d_phase = 2 * np.pi * inst_freq / self.sample_rate
        sig_phase = phase + np.cumsum(d_phase)
        final_phase = sig_phase[-1] % (2 * np.pi)
        
        sig = np.sin(sig_phase)
        
        return sig, final_phase


class LabelGenerator:
    """タイミングシーケンスからフレームレベルのラベルを生成"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def generate_signal_frames(self, timing: List[Tuple[int, int]],
                              num_frames: int) -> np.ndarray:
        """
        タイミングからシグナルフレームを生成
        Returns: SIG_ID_BLANK, SIG_ID_DIT, SIG_ID_DAH, SIG_ID_WORD_SPACE
        """
        signal_frames = np.full(num_frames, SIG_ID_BLANK, dtype=np.int64)
        
        for i in range(num_frames):
            center_sample = i * config.HOP_LENGTH + config.N_FFT // 2
            time_ptr = 0
            for class_id, num_samples in timing:
                if time_ptr <= center_sample < time_ptr + num_samples:
                    if class_id in [SIG_ID_DIT, SIG_ID_DAH]:
                        signal_frames[i] = class_id
                    elif class_id == SIG_ID_WORD_SPACE:
                        signal_frames[i] = SIG_ID_WORD_SPACE
                    else:
                        # INTERNAL_ID_INTRA_GAP や INTER_CHAR は 0 (BACKGROUND) に集約
                        signal_frames[i] = SIG_ID_BLANK
                    break
                time_ptr += num_samples
                if time_ptr > center_sample:
                    break
        
        return signal_frames
    
    def generate_boundary_frames(self, timing: List[Tuple[int, int]],
                                 num_frames: int,
                                 dot_samples: int) -> np.ndarray:
        """
        タイミングからバウンダリフレームを生成
        文字間空白 (INTERNAL_ID_INTER_CHAR) または単語間空白 (SIG_ID_WORD_SPACE) が完了した瞬間のフレームに1.0を立てる
        """
        boundary_frames = np.zeros(num_frames, dtype=np.float32)
        
        time_ptr = 0
        for class_id, num_samples in timing:
            if class_id == INTERNAL_ID_INTER_CHAR:  # 文字の終了 (空白の完了時)
                trigger_sample = time_ptr + num_samples
                trigger_frame = int(trigger_sample // config.HOP_LENGTH)
                for offset in range(5):
                    if 0 <= trigger_frame + offset < num_frames:
                        boundary_frames[trigger_frame + offset] = 1.0
            elif class_id == SIG_ID_WORD_SPACE:  # 単語間空白 (前の文字の終了 + スペース文字の終了)
                # 1. 前の文字の終了境界 (空白開始から 3ユニット分)
                char_end_trigger = time_ptr + int(3 * dot_samples)
                char_end_frame = int(char_end_trigger // config.HOP_LENGTH)
                for offset in range(5):
                    if 0 <= char_end_frame + offset < num_frames:
                        boundary_frames[char_end_frame + offset] = 1.0
                
                # 2. スペース文字自体の終了境界 (空白の終了時点)
                space_end_trigger = time_ptr + num_samples
                space_end_frame = int(space_end_trigger // config.HOP_LENGTH)
                for offset in range(5):
                    if 0 <= space_end_frame + offset < num_frames:
                        boundary_frames[space_end_frame + offset] = 1.0
            
            time_ptr += num_samples
        
        return boundary_frames


class TextGenerator:
    """ランダムテキストやフレーズを生成"""
    
    def generate_random_callsign(self) -> str:
        """リアルなコールサインを生成"""
        prefix_len = random.randint(1, 2)
        prefix = "".join(random.choices(string.ascii_uppercase, k=prefix_len))
        digit = random.choice(string.digits)
        suffix_len = random.randint(1, 3)
        suffix = "".join(random.choices(string.ascii_uppercase, k=suffix_len))
        call = f"{prefix}{digit}{suffix}"
        if random.random() < 0.2:  # Mobile operation
            call += f"/{random.choice(string.digits)}"
        return call
    
    def generate_phrase(self, template: str = None, callsigns: Tuple[str, str] = None) -> str:
        """テンプレートからフレーズを生成"""
        if template is None:
            template = random.choice(config.PHRASE_TEMPLATES)
        
        if callsigns is None:
            callsigns = (self.generate_random_callsign(), self.generate_random_callsign())
        
        call1, call2 = callsigns
        # Generate random strings for name and city to avoid hallucination
        name = "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 6)))
        city = "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 8)))
        weather = random.choice(config.COMMON_WEATHER)
        temp = random.randint(-5, 35)
        rst = f"{random.randint(4, 5)}{random.randint(7, 9)}{random.randint(7, 9)}"
        # 599 -> 5NN conversion for realism
        rst = rst.replace('9', 'N')
        
        phrase = template.format(
            call=call1,
            call1=call1,
            call2=call2,
            name=name,
            city=city,
            rst=rst,
            weather=weather,
            temp=temp
        )
        # ワードの最後にも必ずスペースが入るようにし、挙動を一貫させる
        if not phrase.endswith(" "):
            phrase += " "
        return phrase
    
    def generate_random_tokens(self, available_tokens: List[str],
                              min_len: int, max_len: int,
                              focus_tokens: List[str] = None,
                              focus_prob: float = 0.5,
                              long_gap_prob: float = 0.0,
                              long_gap_duration: float = None) -> List[str]:
        """
        ランダムなトークンリストを生成
        
        Args:
            available_tokens: 使用可能なトークンリスト
            min_len: 最小トークン数
            max_len: 最大トークン数
            focus_tokens: 集中学習するトークン
            focus_prob: focus_tokens を使用する確率
            long_gap_prob: Long Gap を挿入する確率
            long_gap_duration: Long Gap の持続時間（秒）。None の場合はランダムに決定
        
        Returns:
            tokens: トークンリスト（`<GAP:0.5>` のような特殊トークンを含む可能性あり）
        """
        # Randomly choose from available tokens (chars + prosigns)
        valid_tokens = available_tokens if isinstance(available_tokens, list) else list(available_tokens)
        valid_tokens = [t for t in valid_tokens if t != ' ']
        
        # Handle empty valid_tokens
        if not valid_tokens:
            return []
        
        # Determine length
        length = random.randint(min_len, max_len)
        
        # Generate tokens with focus
        if focus_tokens and random.random() < focus_prob:
            focus_valid = [t for t in focus_tokens if t != ' ' and t in valid_tokens]
            if focus_valid:
                k_focus = max(1, length // 2)
                k_other = length - k_focus
                if len(focus_valid) > 1 and k_focus >= len(focus_valid):
                    tokens = random.sample(focus_valid, len(focus_valid))
                    tokens += random.choices(focus_valid, k=k_focus - len(focus_valid))
                else:
                    tokens = random.choices(focus_valid, k=k_focus)
                tokens += random.choices(valid_tokens, k=k_other)
                random.shuffle(tokens)
            else:
                tokens = random.choices(valid_tokens, k=length)
        else:
            tokens = random.choices(valid_tokens, k=length)
        
        # Insert Long Gap tokens
        if long_gap_prob > 0 and random.random() < long_gap_prob:
            # Determine gap duration
            if long_gap_duration is None:
                long_gap_duration = random.uniform(0.5, 2.0)
            
            # Insert at random position (not at start or end)
            if len(tokens) > 2:
                gap_pos = random.randint(1, len(tokens) - 1)
                gap_token = f"<GAP:{long_gap_duration:.2f}>"
                tokens.insert(gap_pos, gap_token)
        
        return tokens
    
    def generate_multiple_phrases(
        self,
        num_phrases: int,
        long_gap_duration: float = None
    ) -> List[str]:
        """
        複数のフレーズを生成し、間に `<GAP:duration>` を挿入
        
        Args:
            num_phrases: 生成するフレーズ数
            long_gap_duration: フレーズ間の Long Gap 持続時間（秒）。None の場合はランダムに決定
        
        Returns:
            tokens: トークンリスト（フレーズ間に `<GAP:duration>` を含む）
        """
        # Generate phrases
        phrases = []
        for _ in range(num_phrases):
            phrase = self.generate_phrase()
            phrases.append(phrase)
        
        # Join with GAP tokens
        if long_gap_duration is None:
            long_gap_duration = random.uniform(0.5, 2.0)
        
        gap_token = f"<GAP:{long_gap_duration:.2f}>"
        
        result = []
        for i, phrase in enumerate(phrases):
            result.append(phrase)
            if i < len(phrases) - 1:
                result.append(gap_token)
        
        return result
    
    def tokens_to_text(self, tokens: List[str]) -> str:
        """トークンリストをテキストに結合"""
        text = "".join(tokens)
        # 空でない場合、末尾にスペースを追加して一貫性を保つ
        if text and not text.endswith(" "):
            text += " "
        return text


class HFChannelSimulator:
    """HFチャネルエフェクトの適用（変更なし）"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def apply_fading(self, waveform: np.ndarray, speed_hz: float = 0.1, min_fading: float = 0.05) -> np.ndarray:
        """Apply Rayleigh-like fading using filtered Gaussian noise."""
        if speed_hz <= 0:
            return waveform
            
        # Generate complex Gaussian noise
        n_samples = len(waveform)
        # Low-pass filter to simulate fading speed (Doppler spread)
        nyquist = 0.5 * self.sample_rate
        b, a = scipy.signal.butter(2, speed_hz / nyquist, btype='low')
        
        # Real and imaginary parts for Rayleigh
        r_real = scipy.signal.lfilter(b, a, np.random.randn(n_samples))
        r_imag = scipy.signal.lfilter(b, a, np.random.randn(n_samples))
        
        # Rayleigh envelope
        fading = np.sqrt(r_real**2 + r_imag**2)
        
        # Normalize fading envelope
        fading /= (np.mean(fading) + 1e-12)
        # Avoid total silence
        fading = np.clip(fading, min_fading, 2.0)
        
        return waveform * fading

    def apply_noise(self, waveform: np.ndarray, impulse_prob: float = 0.0, snr_2500: float = 10.0) -> np.ndarray:
        """Apply AWGN and impulse noise."""

        # AWGN
        # SNR is defined based on the average power of the signal during the MARK (ON) state.
        # For a sine wave with amplitude 1.0, the power is 0.5.
        mark_power = 0.5
        
        # C/N0 (dB-Hz) calculation
        # SNR_2500 = C/N0 - 10*log10(2500)
        cn0_db_hz = snr_2500 + 10 * np.log10(config.SNR_REF_BW)
        
        # Noise power density N0 (Watts/Hz)
        n0 = mark_power / (10**(cn0_db_hz / 10))
        
        # Total noise power in the full bandwidth (Fs/2)
        noise_power = n0 * (self.sample_rate / 2)
        
        noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
        
        # Impulse noise
        impulses = np.zeros(len(waveform))
        if impulse_prob > 0:
            n_impulses = int(len(waveform) * impulse_prob)
            indices = np.random.randint(0, len(waveform), n_impulses)
            impulses[indices] = np.random.uniform(-1, 1, n_impulses)
            
        return waveform + noise + impulses

    def apply_qrm(self, waveform: np.ndarray, snr_2500: float = 5.0) -> np.ndarray:
        """Apply interference from another Morse signal."""

        # Simplified QRM: just another tone with some offset
        t = np.arange(len(waveform)) / self.sample_rate
        offset = random.uniform(-200, 200)
        if abs(offset) < 50: offset = 50 # Avoid exact match
        
        qrm_freq = 700 + offset
        qrm = np.sin(2 * np.pi * qrm_freq * t)
        
        # Simple on-off for QRM
        qrm_mask = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(float)
        qrm *= qrm_mask
        
        sig_avg_watts = np.mean(waveform**2)
        # Use same bandwidth normalization as AWGN for QRM
        cn0_db_hz = snr_2500 + 10 * np.log10(config.SNR_REF_BW)
        n0 = sig_avg_watts / (10**(cn0_db_hz / 10))
        qrm_avg_watts = n0 * (self.sample_rate / 2)
        
        qrm *= np.sqrt(qrm_avg_watts + 1e-12)
        
        return waveform + qrm

    def apply_qrn(self, waveform: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply bursty static crashes (QRN)."""
        n_samples = len(waveform)
        qrn = np.zeros(n_samples)
        # Randomly place 1-5 crashes
        for _ in range(random.randint(1, 5)):
            duration = int(self.sample_rate * random.uniform(0.01, 0.05))
            start = random.randint(0, max(0, n_samples - duration))
            # Bursty noise: white noise multiplied by a window
            burst = np.random.normal(0, strength, duration)
            window = scipy.signal.windows.hann(duration)
            qrn[start:start+duration] += burst * window
        return waveform + qrn

    def apply_out_of_band_qrm(self, waveform: np.ndarray, target_freq: float, strength: float = 2.0) -> np.ndarray:
        """Apply strong signal outside the target filter band to trigger AGC."""
        t = np.arange(len(waveform)) / self.sample_rate
        # Offset significantly from target (e.g., 1-2 kHz away)
        offset = random.choice([-1500, 1500]) + random.uniform(-200, 200)
        qrm_freq = target_freq + offset
        qrm = np.sin(2 * np.pi * qrm_freq * t) * strength
        # Simple on-off pattern
        qrm *= (np.sin(2 * np.pi * 0.3 * t) > 0).astype(float)
        return waveform + qrm

    def apply_agc(self, waveform: np.ndarray, attack_ms: float = 5.0, release_ms: float = 500.0,
                  target_lvl: float = 0.5) -> np.ndarray:
        """
        Simulate Automatic Gain Control (AGC).
        Strong signals/noise reduce gain, which recovers slowly.
        """
        n_samples = len(waveform)
        gain = np.ones(n_samples)
        current_gain = 1.0
        
        # Time constants in samples
        alpha_attack = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000.0))
        alpha_release = np.exp(-1.0 / (release_ms * self.sample_rate / 1000.0))
        
        # Simple envelope follower
        envelope = 0.0
        for i in range(n_samples):
            abs_val = abs(waveform[i])
            if abs_val > envelope:
                envelope = alpha_attack * envelope + (1 - alpha_attack) * abs_val
            else:
                envelope = alpha_release * envelope + (1 - alpha_release) * abs_val
            
            # Gain is inversely proportional to envelope if above target
            if envelope > target_lvl:
                desired_gain = target_lvl / (envelope + 1e-6)
            else:
                desired_gain = 1.0
            
            # Smooth gain changes
            if desired_gain < current_gain: # Attack
                current_gain = alpha_attack * current_gain + (1 - alpha_attack) * desired_gain
            else: # Release
                current_gain = alpha_release * current_gain + (1 - alpha_release) * desired_gain
            
            gain[i] = current_gain
            
        return waveform * gain

    def apply_multipath(self, waveform: np.ndarray, delay_ms: float = 20.0, attenuation: float = 0.5) -> np.ndarray:
        """Apply simple multipath (echo)."""
        delay_samples = int(delay_ms * self.sample_rate / 1000.0)
        if delay_samples >= len(waveform):
            return waveform
        echo = np.zeros_like(waveform)
        echo[delay_samples:] = waveform[:-delay_samples] * attenuation
        return waveform + echo

    def apply_clipping(self, waveform: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Apply non-linear distortion (clipping)."""
        return np.clip(waveform, -threshold, threshold)

    def apply_filter(self, waveform: np.ndarray, center_freq: float = 700.0, bandwidth: float = 500.0) -> np.ndarray:
        """Apply Bandpass filter (Receiver characteristic)."""
        nyquist = 0.5 * self.sample_rate
        low = (center_freq - bandwidth / 2) / nyquist
        high = (center_freq + bandwidth / 2) / nyquist
        b, a = scipy.signal.butter(4, [max(0.01, low), min(0.99, high)], btype='band')
        return scipy.signal.lfilter(b, a, waveform)

    def apply_tx_filter(self, waveform: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply Low-pass filter to simulate TX-side bandwidth limitation (soften edges)."""
        nyquist = 0.5 * self.sample_rate
        # Ensure cutoff is within a reasonable range
        cutoff = np.clip(cutoff, 100, nyquist - 100)
        # Use higher order for more noticeable effect
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='low')
        return scipy.signal.lfilter(b, a, waveform)


class MorseGenerator:
    """既存APIを維持（内部で新しいクラスを委譲）"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.encoder = MorseEncoder()
        self.timing_gen = TimingGenerator(sample_rate)
        self.waveform_gen = WaveformGenerator(sample_rate)
        self.label_gen = LabelGenerator(sample_rate)
    
    # 既存メソッド（内部で新しいクラスを委譲）
    def text_to_morse(self, text: str) -> str:
        return self.encoder.text_to_morse_code(text)
    
    def text_to_morse_tokens(self, text: str) -> List[str]:
        return self.encoder.text_to_tokens(text)
    
    def generate_timing(self, text: str, wpm: int = 20,
                       farnsworth_wpm: int = None, jitter: float = 0.0,
                       weight: float = 1.0, pre_silence: float = None,
                       max_duration: float = config.TRAIN_DURATION) -> List[Tuple[int, int]]:
        return self.timing_gen.generate_timing(text, wpm, farnsworth_wpm, jitter, weight, pre_silence, max_duration)
    
    def estimate_wpm_for_target_frames(self, text: str,
                                       target_frames: int = config.TARGET_FRAMES,
                                       min_wpm: int = 10, max_wpm: int = 45) -> int:
        tokens = self.encoder.text_to_tokens(text)
        return self.timing_gen.estimate_wpm_for_target_frames(tokens, target_frames, min_wpm, max_wpm)
    
    def estimate_max_chars_for_wpm(self, wpm: int,
                                   target_frames: int = config.TARGET_FRAMES) -> int:
        return self.timing_gen.estimate_max_tokens_for_wpm(wpm, target_frames)
    
    def estimate_duration(self, text: str, wpm: int, farnsworth_wpm: int = None, weight: float = 1.0) -> float:
        tokens = self.encoder.text_to_tokens(text)
        return self.timing_gen.estimate_duration(tokens, wpm, farnsworth_wpm, weight)
    
    def reconstruct_text_from_timing(self, timing: List[Tuple[int, int]]) -> str:
        """
        タイミングシーケンス（サンプル数）からテキストを復元する。
        """
        inverse_morse = {v: k for k, v in MORSE_DICT.items()}
        res = ""
        current_code = ""
        
        for class_id, _ in timing:
            if class_id == SIG_ID_DIT:
                current_code += "."
            elif class_id == SIG_ID_DAH:
                current_code += "-"
            elif class_id in [INTERNAL_ID_INTER_CHAR, SIG_ID_WORD_SPACE]: # 文字の区切りまたは単語の区切り
                if current_code:
                    res += inverse_morse.get(current_code, "")
                    current_code = ""
                if class_id == SIG_ID_WORD_SPACE:
                    if not res.endswith(" "):
                        res += " "
            elif class_id == SIG_ID_BLANK: # Silence または GAP
                if current_code:
                    res += inverse_morse.get(current_code, "")
                    current_code = ""
        
        if current_code:
            res += inverse_morse.get(current_code, "")
            
        return res

    def generate_waveform(self, timing: List[Tuple[int, int]],
                          frequency: float = 700.0, waveform_type: str = 'sine',
                          rise_time: float = 0.005, wpm: int = 20,
                          drift_hz: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Waveform generation
        waveform, _ = self.waveform_gen.generate_waveform(
            timing, frequency, waveform_type, rise_time, drift_hz
        )
        
        # Calculate num_frames
        total_samples = len(waveform)
        num_frames = (total_samples - config.N_FFT) // config.HOP_LENGTH + 1
        
        # Generate labels
        signal_frames = self.label_gen.generate_signal_frames(timing, num_frames)
        
        # WPM based dot samples for boundary logic
        dot_samples = int(round((1.2 / wpm) * self.sample_rate))
        boundary_frames = self.label_gen.generate_boundary_frames(timing, num_frames, dot_samples)
        
        return waveform, signal_frames, boundary_frames


def generate_sample(text: str, wpm: int = 20, sample_rate: int = config.SAMPLE_RATE,
                    jitter: float = 0.0, weight: float = 1.0,
                    fading_speed: float = 0.1, min_fading: float = 0.05,
                    frequency: float = 700.0,
                    tx_lowpass: float = None,
                    rise_time: float = 0.005,
                    min_gain_db: float = 0.0,
                    drift_hz: float = 0.0,
                    qrn_strength: float = 0.0,
                    qrm_prob: float = 0.1,
                    impulse_prob: float = 0.001,
                    agc_enabled: bool = False,
                    multipath_delay: float = 0.0,
                    clipping_threshold: float = 1.0,
                    max_duration: float = config.TRAIN_DURATION,
                    snr_2500: float = 10.0) -> Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:

    gen = MorseGenerator(sample_rate=sample_rate)
    sim = HFChannelSimulator(sample_rate=sample_rate)
    
    # Human artifacts are now controlled by arguments
    
    # ワードの最後にも必ずスペースが入るようにし、境界ラベルの挙動を一貫させる
    if not text.endswith(" "):
        text += " "
        
    # Randomly place the signal within the window
    # We estimate the duration to know how much silence we can afford
    total_timing_duration = gen.estimate_duration(text, wpm=wpm, weight=weight)
    available_silence = max_duration - total_timing_duration
    
    safety_margin = 0.1
    if available_silence > safety_margin * 2:
        pre_silence = random.uniform(safety_margin, available_silence - safety_margin)
    else:
        pre_silence = 0.0
        
    # timing を生成 (固定長 max_duration を保証)
    timing = gen.generate_timing(text, wpm=wpm, jitter=jitter, weight=weight,
                                pre_silence=pre_silence, max_duration=max_duration)

    waveform, signal_labels, boundary_labels = gen.generate_waveform(
        timing, frequency=frequency, wpm=wpm, rise_time=rise_time, drift_hz=drift_hz
    )
    
    # Timing からテキストを復元（切り捨て等を正確に反映）
    text = gen.reconstruct_text_from_timing(timing)
    
    # Apply TX filter (soften edges) before channel effects
    if tx_lowpass is not None:
        waveform = sim.apply_tx_filter(waveform, cutoff=tx_lowpass)

    # Only apply channel effects if SNR is below a certain threshold (e.g., 45dB)
    if snr_2500 < 45:
        # 1. Channel propagation effects
        waveform = sim.apply_fading(waveform, speed_hz=fading_speed, min_fading=min_fading)
        if multipath_delay > 0:
            waveform = sim.apply_multipath(waveform, delay_ms=multipath_delay)

        # 2. Add noise and interference (Antenna input)
        waveform = sim.apply_noise(waveform, snr_2500=snr_2500, impulse_prob=impulse_prob)
        if random.random() < qrm_prob:
            waveform = sim.apply_qrm(waveform, snr_2500=snr_2500 + random.uniform(0, 10))
        if qrn_strength > 0:
            waveform = sim.apply_qrn(waveform, strength=qrn_strength)
        if agc_enabled and random.random() < qrm_prob:
            waveform = sim.apply_out_of_band_qrm(waveform, target_freq=frequency, strength=random.uniform(2.0, 5.0))

        # 3. Receiver stage 1: Filtering (Bandpass)
        waveform = sim.apply_filter(waveform, center_freq=frequency)
        
        # 4. Receiver stage 2: AGC (Reacts to filtered signal + remaining noise)
        if agc_enabled:
            waveform = sim.apply_agc(waveform)
            
        # 5. Receiver stage 3: Clipping (Final saturation)
        if clipping_threshold < 1.0:
            waveform = sim.apply_clipping(waveform, threshold=clipping_threshold)
    else:
        # Truly clean: no noise, no filter, no fading
        pass
    
    # Normalize and apply random gain augmentation
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
        # Apply random gain in dB scale if min_gain_db < 0.0
        if min_gain_db < 0.0:
            gain_db = random.uniform(min_gain_db, 0.0)
            gain = 10 ** (gain_db / 20)
            waveform *= gain
        
    return torch.from_numpy(waveform).float(), text, torch.from_numpy(signal_labels).float(), torch.from_numpy(boundary_labels).float()


class CWDataset(Dataset):
    """既存APIを維持（内部で新しいクラスを委譲）"""
    
    def __init__(self, num_samples: int = 1000, min_wpm: int = 15, max_wpm: int = 40,
                 min_snr_2500: float = 10.0, max_snr_2500: float = 30.0,
                 jitter_max: float = 0.1, weight_var: float = 0.2,
                 allowed_chars: str = None, min_len: int = 10,
                 focus_chars: str = None, focus_prob: float = 0.5,
                 fading_speed_min: float = 0.1, fading_speed_max: float = 0.1,
                 min_fading: float = 0.1,
                 phrase_prob: float = 0.0,
                 min_freq: float = 650.0,
                 max_freq: float = 750.0,
                 tx_lowpass_prob: float = 0.8,
                 rise_time_max: float = 0.025,
                 min_gain_db: float = 0.0,
                 gap_prob: float = 0.0,
                 drift_prob: float = 0.0,
                 qrn_prob: float = 0.0,
                 qrm_prob: float = 0.1,
                 impulse_prob: float = 0.001,
                 agc_prob: float = 0.0,
                 multipath_prob: float = 0.0,
                 clipping_prob: float = 0.0):

        self.num_samples = num_samples
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm
        self.min_snr_2500 = min_snr_2500
        self.max_snr_2500 = max_snr_2500
        self.jitter_max = jitter_max
        self.weight_var = weight_var
        self.min_len = min_len
        self.fading_speed_min = fading_speed_min
        self.fading_speed_max = fading_speed_max
        self.min_fading = min_fading
        self.gap_prob = gap_prob
        # config.CHARS includes prosigns now
        self.all_chars = config.CHARS
        self.chars = allowed_chars if allowed_chars else self.all_chars
        self.focus_chars = focus_chars
        self.focus_prob = focus_prob
        self.phrase_prob = phrase_prob
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.tx_lowpass_prob = tx_lowpass_prob
        self.rise_time_max = rise_time_max
        self.min_gain_db = min_gain_db
        self.drift_prob = drift_prob
        self.qrn_prob = qrn_prob
        self.qrm_prob = qrm_prob
        self.impulse_prob = impulse_prob
        self.agc_prob = agc_prob
        self.multipath_prob = multipath_prob
        self.clipping_prob = clipping_prob
        self.text_gen = TextGenerator()
        self.morse_gen = MorseGenerator()
        self.channel_sim = HFChannelSimulator()

    def generate_random_callsign(self) -> str:
        """Generate a realistic random callsign."""
        return self.text_gen.generate_random_callsign()

    def generate_phrase(self) -> str:
        """Generate a text from templates."""
        return self.text_gen.generate_phrase()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        max_duration = config.TRAIN_DURATION
        # Filling Rate: 0.7 〜 1.0 の割合でバッファを埋める
        filling_rate = random.uniform(0.7, 1.0)
        target_duration = max_duration * filling_rate

        is_phrase = random.random() < self.phrase_prob
        
        for attempt in range(5):  # Retry if text is too long for WPM limits
            if is_phrase:
                # 目標時間を埋めるまでフレーズを連結
                current_text = ""
                while True:
                    new_phrase = self.generate_phrase()
                    # 連結した場合の長さを推定 (max_wpm で収まるかチェック)
                    test_text = current_text + (" " if current_text else "") + new_phrase
                    # 推定時間は GAP を考慮しないため、少し余裕を持つ
                    if self.morse_gen.estimate_duration(test_text, wpm=self.max_wpm) > target_duration * 1.1:
                        break
                    current_text = test_text
                    
                    # フレーズ間に GAP を挿入する判定
                    if self.gap_prob > 0 and random.random() < self.gap_prob:
                        gap_sec = random.uniform(1.0, 3.0)
                        current_text += f" <GAP:{gap_sec:.1f}>"
                        # GAP を入れた直後は流石に次はフレーズを入れるか、あるいは終了判定へ
                        if self.morse_gen.estimate_duration(current_text, wpm=self.max_wpm) > target_duration:
                            break
                
                text = current_text if current_text else self.generate_phrase()

                # WPMを決めて文字数を制限する
                # フレーズの場合は、target_duration に収まるような WPM を逆算する (Adaptive)
                # ただし、filling_rate が低い場合でも極端に遅くならないよう min_wpm は守る
                wpm = self.morse_gen.estimate_wpm_for_target_frames(
                    text, 
                    target_frames=int(target_duration * config.SAMPLE_RATE / config.HOP_LENGTH),
                    min_wpm=self.min_wpm, 
                    max_wpm=self.max_wpm
                )
                
                # 推定された WPM でもはみ出す場合は切り捨てる（特にフレーズが長すぎる場合）
                # ここでは正確な長さ計算よりも、単に長すぎる場合の後方カットを行う
                max_allowed_len = self.morse_gen.estimate_max_chars_for_wpm(wpm, target_frames=int(target_duration * config.SAMPLE_RATE / config.HOP_LENGTH))
                phrase_tokens = self.morse_gen.text_to_morse_tokens(text)
                if len(phrase_tokens) > max_allowed_len:
                     # GAP トークンを保護しつつ切り捨て
                    text = "".join(phrase_tokens[:max_allowed_len])
                
                # Verify if it fits (物理時間の再確認)
                timing = self.morse_gen.generate_timing(text, wpm=wpm)
                total_time_sec = sum(t[1] for t in timing) / config.SAMPLE_RATE
                
                # max_duration を超えていなければ OK
                # filling_rate による target_duration はあくまで目安（短くてもOK）だが、長すぎて溢れるのはNG
                if total_time_sec <= max_duration:
                    break
                else:
                    # 溢れた場合はリトライ（attemptが進む）
                    if wpm >= self.max_wpm and attempt < 4:
                        continue
                    break
            else:
                # ランダム生成: WPM を先に決定 (正規分布中心)
                wpm_center = 20
                wpm_sigma = (self.max_wpm - self.min_wpm) / 4
                wpm = int(random.gauss(wpm_center, wpm_sigma))
                wpm = max(self.min_wpm, min(self.max_wpm, wpm))

                # 指定 WPM と target_duration で入る文字数を計算
                max_allowed_len = self.morse_gen.estimate_max_chars_for_wpm(
                    wpm, 
                    target_frames=int(target_duration * config.SAMPLE_RATE / config.HOP_LENGTH)
                )
                
                # ランダム文字列生成
                # target_limit は filling_rate に従った文字数
                target_limit = max(self.min_len, max_allowed_len - 1)
                
                # Focus chars (50% chance if specified)
                valid_chars = self.chars
                if self.focus_chars and random.random() < self.focus_prob:
                     valid_chars = self.focus_chars
                
                # Correctly tokenize the valid_chars if it's a string, otherwise use as list
                if isinstance(valid_chars, str):
                    valid_tokens = self.morse_gen.text_to_morse_tokens(valid_chars)
                else:
                    valid_tokens = list(valid_chars)
                
                valid_tokens = [t for t in valid_tokens if t != ' '] # Exclude spaces from random pool
                
                length = random.randint(self.min_len, target_limit)

                # Prioritize numbers or focus chars if needed (basic logic kept)
                # But here we simply choose from valid_tokens
                if self.focus_chars and valid_chars == self.focus_chars:
                    # If focusing, force at least some focus chars
                    num_focus = length // 2
                    num_others = length - num_focus
                    
                    # Convert to token list correctly (handling Prosigns like <NJ>)
                    focus_list = self.morse_gen.text_to_morse_tokens(self.focus_chars) if isinstance(self.focus_chars, str) else list(self.focus_chars)
                    focus_list = [t for t in focus_list if t != ' ']
                    
                    chars_list = self.morse_gen.text_to_morse_tokens(self.chars) if isinstance(self.chars, str) else list(self.chars)
                    chars_list = [t for t in chars_list if t != ' ']

                    tokens = random.choices(focus_list, k=num_focus)
                    tokens += random.choices(chars_list, k=num_others)
                    random.shuffle(tokens)
                else:
                    tokens = random.choices(valid_tokens, k=length)
                
                # ランダムに 0〜1 箇所の GAP を挿入 (1.0秒以上)
                if self.gap_prob > 0 and random.random() < self.gap_prob:
                    gap_sec = random.uniform(1.0, 5.0)
                    gap_token = f"<GAP:{gap_sec:.1f}>"
                    # トークンリストの任意の位置（0〜末尾）に挿入
                    pos = random.randint(0, len(tokens))
                    tokens.insert(pos, gap_token)

                # Join them, ensuring the final token count (including spaces) does not exceed target_limit
                text = ""
                token_count = 0
                for t in tokens:
                    # GAP トークンは文字数制限 (target_limit) にカウントせず、そのまま追加する
                    if t.startswith("<GAP:"):
                        text += t
                        continue
                    
                    if token_count >= target_limit: break
                    text += t
                    token_count += 1
                    if token_count < target_limit and random.random() < 0.4:
                        text += " "
                        token_count += 1
        # カリキュラム設定に基づいて SNR を決定
        # 指定された [min, max] の範囲に 90% のサンプルが収まるような正規分布を使用する。
        # 正規分布の 90% 信頼区間は mu +/- 1.645 * sigma である。
        mu = (self.min_snr_2500 + self.max_snr_2500) / 2
        sigma = (self.max_snr_2500 - self.min_snr_2500) / (2 * 1.645)
        snr = random.normalvariate(mu, max(sigma, 1e-6))
        
        # Determine jitter and weight based on curriculum settings
        jitter = random.uniform(0, self.jitter_max)
        # weight is centered at 1.0, variation is +/- weight_var
        weight = 1.0 + random.uniform(-self.weight_var, self.weight_var)
        
        fading_speed = random.uniform(self.fading_speed_min, self.fading_speed_max)
        
        frequency = random.uniform(self.min_freq, self.max_freq)

        # Randomly apply TX lowpass filter to soften edges
        tx_lowpass = None
        rise_time = 0.005  # Default
        if random.random() < self.tx_lowpass_prob:
            # Cutoff is typically somewhere above the carrier frequency.
            # 0.8x to 2.5x frequency covers from "muffled" to "standard".
            tx_lowpass = frequency * random.uniform(0.8, 2.5)
            # Also soften the rise time itself
            rise_time = random.uniform(0.005, self.rise_time_max)
        
        # New Augmentations based on dataset probabilities (set by curriculum)
        drift_hz = 0.0
        if random.random() < self.drift_prob:
            drift_hz = random.uniform(1.0, 15.0)
            
        qrn_strength = 0.0
        if random.random() < self.qrn_prob:
            # Strength relative to signal (mark_power=0.5)
            qrn_strength = random.uniform(0.5, 3.0)
            
        agc_enabled = random.random() < self.agc_prob
        
        multipath_delay = 0.0
        if random.random() < self.multipath_prob:
            multipath_delay = random.uniform(10.0, 50.0)
            
        clipping_threshold = 1.0
        if random.random() < self.clipping_prob:
            clipping_threshold = random.uniform(0.3, 0.8)

        waveform, label, signal_labels, boundary_labels = generate_sample(
            text, wpm=wpm, snr_2500=snr, jitter=jitter, weight=weight,
            fading_speed=fading_speed, min_fading=self.min_fading,
            frequency=frequency,
            tx_lowpass=tx_lowpass,
            rise_time=rise_time,
            min_gain_db=self.min_gain_db,
            drift_hz=drift_hz,
            qrn_strength=qrn_strength,
            qrm_prob=self.qrm_prob,
            impulse_prob=self.impulse_prob,
            agc_enabled=agc_enabled,
            multipath_delay=multipath_delay,
            clipping_threshold=clipping_threshold
        )
        # Return wpm as well so the trainer can use it for adaptive space reconstruction
        return waveform, label, wpm, signal_labels, boundary_labels, is_phrase


if __name__ == "__main__":
    sample_text = "CQ DE KILO CODE K"
    sample_rate = config.SAMPLE_RATE
    print(f"Generating sample: {sample_text}")
    
    waveform, label, signal_labels, _ = generate_sample(sample_text, wpm=25, snr_2500=20, sample_rate=sample_rate)
    
    output_file = "sample_cw.wav"
    # Convert back to numpy for scipy saving
    wf_np = waveform.numpy()
    scipy.io.wavfile.write(output_file, sample_rate, (wf_np * 32767).astype(np.int16))
    
    print(f"Saved to {output_file} using scipy")
    
    # Test Dataset
    dataset = CWDataset(num_samples=5)
    wf, lbl, wpm, sig = dataset[0]
    print(f"Dataset test - Waveform shape: {wf.shape}, Label: {lbl}, Signal shape: {sig.shape}")
