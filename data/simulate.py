import matplotlib.pyplot as plt
from .pattern_model import PatternModel

def main(use_real=False):
    model = PatternModel(use_real_data=use_real)

    if use_real:
        t, heartbeat, domains, metadata = model.use_data_model()
        plt.figure(figsize=(10, 5))
        plt.plot(t, heartbeat, label="Cultural Heartbeat (Real Data)", linewidth=2)
        plt.title("Secular Pendulum — Data-Driven Mode")
        plt.xlabel("Index / Time")
        plt.ylabel("Order ↔ Novelty Axis")
        plt.legend()
        plt.grid(True)
    else:
        t, signal = model.generate_trends()
        plt.figure(figsize=(10, 5))
        plt.plot(t, signal, label="Theoretical Oscillation", linewidth=2)
        plt.title("Secular Pendulum — Theoretical Mode")
        plt.xlabel("Time (Decades)")
        plt.ylabel("Cultural Motion (Order ↔ Novelty)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Switch between modes here
    main(use_real=True)

