import pandas as pd
import numpy as np

def load_trend_data(path: str = "data/trends_seed.csv"):
    """
    Loads trend data and extracts numeric features for modeling.
    Works with mixed text + numeric CSVs (like your trend dataset).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Normalize numeric columns to [-1, 1]
    normalized = {}
    for col in numeric_cols:
        series = df[col].fillna(0).to_numpy(dtype=float)
        if np.max(series) != np.min(series):
            norm = 2 * ((series - np.min(series)) / (np.max(series) - np.min(series))) - 1
        else:
            norm = series
        normalized[col] = norm

    norm_df = pd.DataFrame(normalized)

    # Attach metadata columns if present
    for col in ["timestamp", "trend_id", "domain", "text_blurb"]:
        if col in df.columns:
            norm_df[col] = df[col]

    return norm_df


def get_cultural_heartbeat(df: pd.DataFrame):
    """
    Computes the overall Order↔Novelty cultural heartbeat
    from normalized numeric signals.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found for heartbeat calculation.")
    return df[numeric_cols].mean(axis=1).to_numpy()


def prepare_features():
    """
    Returns (time, numeric_data, heartbeat, metadata)
    — used by the model training and visualization layers.
    """
    df = load_trend_data()
    heartbeat = get_cultural_heartbeat(df)
    time = np.arange(len(df))

    metadata = df[[c for c in ["timestamp", "trend_id", "domain", "text_blurb"] if c in df.columns]]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = {col: df[col].to_numpy() for col in numeric_cols}

    return time, numeric_data, heartbeat, metadata


if __name__ == "__main__":
    # --- Load, expand, and label the dataset ---
    df = load_trend_data()

    # 🔹 Expand dataset (create synthetic variations for training)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    synthetic = []
    for _ in range(200):
        new = df.copy()
        for c in numeric_cols:
            new[c] += np.random.normal(0, 0.1, size=len(new))
        synthetic.append(new)
    df = pd.concat(synthetic, ignore_index=True)

    # 🔹 Derive axis_label (-1 = Order, +1 = Novelty)
    df["axis_label"] = np.sign(df[numeric_cols].mean(axis=1))

    # 🔹 Save the expanded features
    df.to_csv("data/features.csv", index=False)
    print(f"✅ Saved expanded dataset with {len(df)} rows and axis_label column.")
