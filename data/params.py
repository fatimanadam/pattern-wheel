import numpy as np

# === PERIODS (in months) ===
T1 = 192           # ~16 years primary Order/Novelty swing
T_RES = 120        # 10-year resonance
T_FOURTH = 1056    # 88-year Fourth Turning macro cycle

# === AMPLITUDE & PHASE ===
A1, A2 = 0.7, 0.3
PHI1, PHI2 = 0.0, 0.0
E1, PHI_M = 0.25, 0.0

# === SCARCITY (Need-for-Opposite) ===
EMA_WIN = 24
K_SCARCITY = 0.35

# === YOUTH INFLUENCE ===
Y_LEVEL = 0.6
GAMMA = 0.5

# === DRIVER STRENGTH ===
BETA = 0.6

# === LAGS & NOISE ===
L_DEFAULT = 6
NOISE_SIGMA = 0.03
RNG_SEED = 42

# === DOMAINS ===
DOMAIN_NAMES = ["fashion", "music", "tech", "politics", "economics"]

# === COUPLING MATRICES ===
A = np.array([
 [0.40, 0.06, 0.08, 0.00, 0.00],  # fashion <= music, tech
 [0.05, 0.40, 0.05, 0.00, 0.00],  # music <= fashion, tech
 [0.03, 0.05, 0.45, 0.00, 0.00],  # tech <= fashion, music
 [0.00, 0.00, 0.07, 0.45, 0.12],  # politics <= tech, econ
 [0.00, 0.00, 0.08, -0.04, 0.45], # econ <= tech, -politics
])

A_LAG = np.array([
 [0.00, 0.08, 0.07, -0.05, 0.00],
 [0.00, 0.00, 0.03, 0.00, 0.00],
 [0.00, 0.00, 0.00, 0.00, 0.00],
 [0.00, 0.00, 0.06, 0.00, 0.00],
 [0.00, 0.00, 0.04, 0.03, 0.00],
])
