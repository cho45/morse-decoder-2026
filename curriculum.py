import config
from data_gen import MORSE_DICT

def get_prefixes(char):
    code = MORSE_DICT.get(char, "")
    prefixes = []
    for i in range(1, len(code)):
        prefix_code = code[:i]
        # Find char with this code
        for c, m in MORSE_DICT.items():
            if m == prefix_code:
                prefixes.append(c)
                break
    return prefixes

# Build Curriculum Sets starting from numbers 1-0
# For each number, we add it and all its prefixes.
NUMBERS = "1234567890"
CURRICULUM_SETS = []
seen_chars = set()

for num in NUMBERS:
    current_set = [num]
    prefixes = get_prefixes(num)
    for p in prefixes:
        if p not in seen_chars:
            current_set.append(p)
    
    # Add to curriculum
    CURRICULUM_SETS.append("".join(current_set))
    seen_chars.update(current_set)

# Add remaining characters
# [修正] スペース ' ' は物理的な空白として Signal Head で扱うため、
# カリキュラムの「文字」リストからは除外する。
REMAINING = "KLPQXYZCF/?.,-()'!&:;=+_\"$@"
REMAINING = REMAINING.replace(" ", "")
for char in REMAINING:
    if char not in seen_chars:
        CURRICULUM_SETS.append(char)
        seen_chars.add(char)

# Add prosigns to curriculum
for ps in config.PROSIGNS:
    if ps not in seen_chars:
        CURRICULUM_SETS.append(ps)
        seen_chars.add(ps)

class CurriculumPhase:
    def __init__(self, name, chars, focus_chars=None, min_snr_2500=100.0, max_snr_2500=100.0, min_wpm=20, max_wpm=20,
                 jitter=0.0, weight_var=0.0, phrase_prob=0.0, focus_prob=0.5,
                 fading_speed=(0.0, 0.0), min_fading=1.0,
                 drift_prob=0.0, qrn_prob=0.0, qrm_prob=0.1, impulse_prob=0.001,
                 agc_prob=0.0, multipath_prob=0.0, clipping_prob=0.0, min_gain_db=0.0,
                 penalty_weight=2.0):
        self.name = name
        self.chars = chars
        self.focus_chars = focus_chars
        self.min_snr_2500 = min_snr_2500
        self.max_snr_2500 = max_snr_2500
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm
        self.jitter = jitter
        self.weight_var = weight_var
        self.phrase_prob = phrase_prob
        self.focus_prob = focus_prob
        self.fading_speed = fading_speed
        self.min_fading = min_fading
        self.drift_prob = drift_prob
        self.qrn_prob = qrn_prob
        self.qrm_prob = qrm_prob
        self.impulse_prob = impulse_prob
        self.agc_prob = agc_prob
        self.multipath_prob = multipath_prob
        self.clipping_prob = clipping_prob
        self.min_gain_db = min_gain_db
        self.penalty_weight = penalty_weight

class CurriculumManager:
    def __init__(self):
        self.phases = []
        self._build_phases()

    def _build_phases(self):
        # 1. Character Introduction Phases (Clean environment)
        current_chars = ""
        for i, s in enumerate(CURRICULUM_SETS):
            current_chars += s
            self.phases.append(CurriculumPhase(
                name=f"Char_{i+1}_{s}",
                chars=current_chars,
                focus_chars=s,
                min_snr_2500=100.0, max_snr_2500=100.0,
                min_wpm=15, max_wpm=25,
                min_gain_db=-20,
                focus_prob=0.7, # High focus on new chars
                penalty_weight=3.0 # Disciplined but not crushing
            ))

        # 2. Environmental Degradation Phases
        max_chars = current_chars
        
        # Slight Variations A
        self.phases.append(CurriculumPhase(
            name="Slight_Var_A", chars=max_chars,
            min_snr_2500=30.0, max_snr_2500=45.0, min_wpm=15, max_wpm=30,
            jitter=0.015, weight_var=0.025, phrase_prob=0.3, min_gain_db=-30,
            penalty_weight=3.0
        ))

        # Slight Variations B
        self.phases.append(CurriculumPhase(
            name="Slight_Var_B", chars=max_chars,
            min_snr_2500=30.0, max_snr_2500=45.0, min_wpm=15, max_wpm=30,
            jitter=0.03, weight_var=0.05, phrase_prob=0.3, min_gain_db=-30,
            penalty_weight=4.0
        ))

        # Practical 1 (Fading Only)
        self.phases.append(CurriculumPhase(
            name="Practical_1", chars=max_chars,
            min_snr_2500=20.0, max_snr_2500=30.0, min_wpm=15, max_wpm=35,
            fading_speed=(0.0, 0.1), min_fading=0.4, phrase_prob=0.5, min_gain_db=-40,
            penalty_weight=3.0
        ))

        # Practical 1 + Drift
        self.phases.append(CurriculumPhase(
            name="Practical_1_Drift", chars=max_chars,
            min_snr_2500=20.0, max_snr_2500=30.0, min_wpm=15, max_wpm=35,
            fading_speed=(0.0, 0.1), min_fading=0.4, drift_prob=0.2, phrase_prob=0.5, min_gain_db=-40,
            penalty_weight=3.0
        ))

        # Practical 1 + Drift + AGC
        self.phases.append(CurriculumPhase(
            name="Practical_1_AGC", chars=max_chars,
            min_snr_2500=20.0, max_snr_2500=30.0, min_wpm=15, max_wpm=35,
            fading_speed=(0.0, 0.1), min_fading=0.4, drift_prob=0.2, agc_prob=0.2, phrase_prob=0.5, min_gain_db=-40,
            penalty_weight=3.0
        ))

        # Practical 2 (SNR Reduction Stage 1)
        self.phases.append(CurriculumPhase(
            name="Practical_2", chars=max_chars,
            min_snr_2500=13.0, max_snr_2500=23.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.1), min_fading=0.5, drift_prob=0.3, agc_prob=0.3, phrase_prob=0.5, min_gain_db=-50.0,
            penalty_weight=1.0
        ))

        # Practical 3 (SNR Reduction Stage 2)
        self.phases.append(CurriculumPhase(
            name="Practical_3", chars=max_chars,
            min_snr_2500=7.0, max_snr_2500=17.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.1), min_fading=0.6, drift_prob=0.3, agc_prob=0.4, phrase_prob=0.5, min_gain_db=-60.0,
            penalty_weight=0.5
        ))

        # Negative SNR 1
        self.phases.append(CurriculumPhase(
            name="Negative_1", chars=max_chars,
            min_snr_2500=1.0, max_snr_2500=11.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.05), min_fading=0.7, drift_prob=0.2, agc_prob=0.4, phrase_prob=0.5, qrn_prob=0.2, min_gain_db=-60.0,
            penalty_weight=0.3
        ))

        # Negative SNR 2 (Deeper Noise)
        self.phases.append(CurriculumPhase(
            name="Negative_2", chars=max_chars,
            min_snr_2500=-5.0, max_snr_2500=5.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.02), min_fading=0.8, drift_prob=0.1, agc_prob=0.3, phrase_prob=0.5, qrn_prob=0.3, clipping_prob=0.2, min_gain_db=-60.0,
            penalty_weight=0.2
        ))

        # 処理利得: 2500Hz 帯域から 31.25Hz ビンへの絞り込みにより、+19.0 dB の利得が発生します。
        # 限界値の定義: SNR_2500 = -18 dB において、ビン内 SNR は +1.0 dB となり、信号がノイズをわずかに上回る物理的な検知限界点となります。
        self.phases.append(CurriculumPhase(
            name="Extreme", chars=max_chars,
            min_snr_2500=-17, max_snr_2500=0.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.02), min_fading=0.9, drift_prob=0.1, agc_prob=0.2, phrase_prob=0.5, qrn_prob=0.3, clipping_prob=0.2, min_gain_db=-60.0,
            penalty_weight=0.1
        ))

    def get_phase(self, phase_idx):
        idx = max(0, min(phase_idx - 1, len(self.phases) - 1))
        return self.phases[idx]

    def get_max_phase(self):
        return len(self.phases)

if __name__ == "__main__":
    cm = CurriculumManager()
    print(f"Total phases: {cm.get_max_phase()}")
    print(f"{'No':>3} | {'Name':<20} | {'SNR':<12} | {'WPM':<10} | {'Chars'}")
    print("-" * 80)
    for i in range(1, cm.get_max_phase() + 1):
        p = cm.get_phase(i)
        snr_range = f"{p.min_snr_2500:.1f}~{p.max_snr_2500:.1f}"
        wpm_range = f"{p.min_wpm}~{p.max_wpm}"
        print(f"{i:>3} | {p.name:<20} | {snr_range:<12} | {wpm_range:<10} | {p.chars}")