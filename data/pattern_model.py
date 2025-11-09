import numpy as np
from .build_features import prepare_features

class PatternModel:
    """
    The Secular Pendulum — models cyclical movement between Order and Novelty
    across cultural domains using dual-frequency oscillations.
    Can simulate theory-driven cycles or load real trend data.
    """

    def __init__(self, amplitude1=0.3, amplitude2=0.6, scarcity=0.4, youth_weight=0.5,
                 coupling=0.6, use_real_data=False):
        self.a1 = amplitude1
        self.a2 = amplitude2
        self.k = scarcity
        self.y = youth_weight
        self.beta = coupling
        self.use_real_data = use_real_data

        if self.use_real_data:
            self.time, self.domains, self.heartbeat, self.metadata = prepare_features()

    # ------------------------------------------------------------------
    # 1. THEORETICAL OSCILLATION MODEL
    # ------------------------------------------------------------------
    def oscillate(self, t):
        base_wave = self.a1 * np.sin(t / 6)
        youth_wave = self.a2 * np.sin(t / 2 + self.y * np.pi)
        scarcity_effect = np.exp(-self.k * np.abs(base_wave))
        return (base_wave + self.beta * youth_wave) * scarcity_effect

    def generate_trends(self, length=960, shock_month=400):
        t = np.arange(0, length / 12, 0.1)
        signal = self.oscillate(t)
        shock_effect = np.zeros_like(signal)
        shock_effect[(t > shock_month / 12) & (t < (shock_month / 12 + 1))] = 0.3
        signal += shock_effect
        return t, signal

    # ------------------------------------------------------------------
    # 2. DATA-DRIVEN MODE
    # ------------------------------------------------------------------
    def use_data_model(self):
        if not self.use_real_data:
            raise ValueError("This model was initialized without use_real_data=True.")
        return self.time, self.heartbeat, self.domains, self.metadata

    # ------------------------------------------------------------------
    # 3. SUMMARY
    # ------------------------------------------------------------------
    def get_data_summary(self):
        if not self.use_real_data:
            return {"mode": "synthetic", "description": "Using mathematical oscillation model."}
        return {
            "mode": "real-data",
            "num_entries": len(self.time),
            "domains": list(self.domains.keys()),
            "mean_heartbeat": float(np.mean(self.heartbeat))
        }


if __name__ == "__main__":
    model = PatternModel(use_real_data=True)
    print(model.get_data_summary())
