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

class CurriculumPhase:
    def __init__(self, name, chars, min_snr=100.0, max_snr=100.0, min_wpm=20, max_wpm=20,
                 jitter=0.0, weight_var=0.0, phrase_prob=0.0, focus_prob=0.5,
                 fading_speed=(0.0, 0.0), min_fading=1.0,
                 drift_prob=0.0, qrn_prob=0.0, qrm_prob=0.1, impulse_prob=0.001,
                 agc_prob=0.0, multipath_prob=0.0, clipping_prob=0.0, min_gain_db=0.0,
                 penalty_weight=2.0):
        self.name = name
        self.chars = chars
        self.min_snr = min_snr
        self.max_snr = max_snr
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
                min_snr=100.0, max_snr=100.0,
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
            min_snr=25.0, max_snr=40.0, min_wpm=15, max_wpm=30,
            jitter=0.015, weight_var=0.025, phrase_prob=0.3, min_gain_db=-30,
            penalty_weight=3.0
        ))

        # Slight Variations B
        self.phases.append(CurriculumPhase(
            name="Slight_Var_B", chars=max_chars,
            min_snr=25.0, max_snr=40.0, min_wpm=15, max_wpm=30,
            jitter=0.03, weight_var=0.05, phrase_prob=0.3, min_gain_db=-30,
            penalty_weight=4.0
        ))

        # Practical 1 (Fading Only)
        self.phases.append(CurriculumPhase(
            name="Practical_1", chars=max_chars,
            min_snr=15.0, max_snr=25.0, min_wpm=15, max_wpm=35,
            fading_speed=(0.0, 0.1), min_fading=0.4, phrase_prob=0.5, min_gain_db=-40,
            penalty_weight=3.0
        ))

        # Practical 1 + Drift
        self.phases.append(CurriculumPhase(
            name="Practical_1_Drift", chars=max_chars,
            min_snr=15.0, max_snr=25.0, min_wpm=15, max_wpm=35,
            fading_speed=(0.0, 0.1), min_fading=0.4, drift_prob=0.2, phrase_prob=0.5, min_gain_db=-40,
            penalty_weight=3.0
        ))

        # Practical 1 + Drift + AGC
        self.phases.append(CurriculumPhase(
            name="Practical_1_AGC", chars=max_chars,
            min_snr=15.0, max_snr=25.0, min_wpm=15, max_wpm=35,
            fading_speed=(0.0, 0.1), min_fading=0.4, drift_prob=0.2, agc_prob=0.2, phrase_prob=0.5, min_gain_db=-40,
            penalty_weight=3.0
        ))

        # Practical 2 (SNR Reduction Stage 1)
        self.phases.append(CurriculumPhase(
            name="Practical_2", chars=max_chars,
            min_snr=8.0, max_snr=18.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.1), min_fading=0.5, drift_prob=0.3, agc_prob=0.3, phrase_prob=0.5, min_gain_db=-50.0,
            penalty_weight=1.0
        ))

        # Practical 3 (SNR Reduction Stage 2)
        self.phases.append(CurriculumPhase(
            name="Practical_3", chars=max_chars,
            min_snr=2.0, max_snr=12.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.1), min_fading=0.6, drift_prob=0.3, agc_prob=0.4, phrase_prob=0.5, min_gain_db=-60.0,
            penalty_weight=0.5
        ))

        # Negative SNR 1
        self.phases.append(CurriculumPhase(
            name="Negative_1", chars=max_chars,
            min_snr=-4.0, max_snr=6.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.05), min_fading=0.7, drift_prob=0.2, agc_prob=0.4, phrase_prob=0.5, qrn_prob=0.2, min_gain_db=-60.0,
            penalty_weight=0.3
        ))

        # Negative SNR 2 (Deeper Noise)
        self.phases.append(CurriculumPhase(
            name="Negative_2", chars=max_chars,
            min_snr=-10.0, max_snr=0.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.02), min_fading=0.8, drift_prob=0.1, agc_prob=0.3, phrase_prob=0.5, qrn_prob=0.3, clipping_prob=0.2, min_gain_db=-60.0,
            penalty_weight=0.2
        ))

        # Extreme SNR (The Ultimate Challenge)
        # SNR -15dB. Fading and other distortions are minimized to focus on pure noise robustness.
        self.phases.append(CurriculumPhase(
            name="Extreme", chars=max_chars,
            min_snr=-15.0, max_snr=-5.0, min_wpm=15, max_wpm=40,
            fading_speed=(0.0, 0.0), min_fading=1.0, drift_prob=0.0, agc_prob=0.0, phrase_prob=0.5, qrn_prob=0.4, clipping_prob=0.3, min_gain_db=-60.0,
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
    for i in range(1, cm.get_max_phase() + 1):
        p = cm.get_phase(i)
        print(f"Phase {i}: {p.name} | Chars: {p.chars}")