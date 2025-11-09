import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path

# These should match the features built in your build_features.py
FEATURES = [
    "gt_search",
    "yt_views",
    "sp500_ret",
    "cpi_surprise",
    "unemp_rate",
    "youth_proxy",
    "shock_signed",
    "novelty_kw_density",
    "order_kw_density",
]

def load_and_build(csv_path="data/features.csv"):
    """
    Loads the normalized features dataset for model training.
    """
    df = pd.read_csv(csv_path)
    return df

def main():
    df = load_and_build()
    df = df.dropna(subset=["axis_label"])

    # Construct the feature matrix (X) and target vector (y)
    lag_cols = [
        f"{c}_lag{L}"
        for c in ["gt_search", "tiktok_views", "youth_proxy", "shock_signed"]
        for L in (1, 3, 6)
    ]
    available_lags = [col for col in lag_cols if col in df.columns]
    feature_cols = [col for col in FEATURES + available_lags if col in df.columns]

    X = df[feature_cols].values
    y = df["axis_label"].values

    # Simple random 80/20 train-test split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )

    # Train Ridge regression model
    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    # Evaluate performance
    print("R²:", r2_score(y[test_idx], pred))
    print("MAE:", mean_absolute_error(y[test_idx], pred))

    # Save trained model and metadata
    Path("backend").mkdir(exist_ok=True)
    joblib.dump(model, "backend/axis_model.pkl")
    meta = {"features": feature_cols}
    joblib.dump(meta, "backend/axis_model_meta.pkl")

    print("✅ Model training complete.")
    print("Saved model to backend/axis_model.pkl and metadata to backend/axis_model_meta.pkl.")

if __name__ == "__main__":
    main()
