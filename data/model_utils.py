import numpy as np
import joblib
import re

# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------
def load_model(path="backend/axis_model.pkl"):
    """Load the trained Ridge regression model."""
    return joblib.load(path)


# ---------------------------------------------------------
# 2. Axis prediction helper
# ---------------------------------------------------------
def predict_axis(model, feature_dict):
    """Predict cultural axis (Order ↔ Novelty) using the model."""
    features = np.array(list(feature_dict.values())).reshape(1, -1)
    prediction = model.predict(features)[0]
    label = "Novelty" if prediction >= 0 else "Order"
    confidence = min(1.0, abs(prediction))  # pseudo-confidence
    return label, confidence, prediction


# ---------------------------------------------------------
# 3. Scenario parser (for chatbot input)
# ---------------------------------------------------------
def parse_scenario(user_text):
    """Extract simple meaning from user text."""
    text = user_text.lower()
    keywords = []

    if "makeup" in text or "beauty" in text:
        keywords.append("Makeup & Beauty trends")
    if "fashion" in text or "style" in text:
        keywords.append("Fashion cycles")
    if "tech" in text or "ai" in text:
        keywords.append("Technology evolution")
    if "music" in text:
        keywords.append("Music culture")
    if "economy" in text or "recession" in text or "inflation" in text:
        keywords.append("Economic influence")

    if not keywords:
        return "General cultural and social dynamics"
    return ", ".join(keywords)


# ---------------------------------------------------------
# 4. Forecast generation
# ---------------------------------------------------------
def generate_forecast(model, base_features, scenario_text, months=60):
    """Simulate how culture evolves over time."""
    # Adjust base features based on scenario
    f = base_features.copy()
    t = np.linspace(0, months, 200)

    if any(word in scenario_text.lower() for word in ["crisis", "recession", "inflation", "scarcity"]):
        f["sp500_ret"] = min(1.0, f["sp500_ret"] + 0.3)
    if any(word in scenario_text.lower() for word in ["youth", "gen z", "trend", "viral"]):
        f["youth_proxy"] = 0.8
    if any(word in scenario_text.lower() for word in ["innovation", "ai", "tech", "future", "digital"]):
        f["yt_views"] = 0.8
    if any(word in scenario_text.lower() for word in ["nostalgia", "tradition", "vintage"]):
        f["order_kw_density"] = 0.8
        f["novelty_kw_density"] = -0.3

    # Simulate axis curve
    base_amp = f["yt_views"] - f["order_kw_density"]
    signal = (
        0.5 * np.sin(2 * np.pi * t / 24)
        + 0.3 * np.sin(4 * np.pi * t / 24)
        + base_amp * np.exp(-t / months)
    )

    # Add model-based adjustment
    y_pred = model.predict(np.tile(np.array(list(f.values())), (len(t), 1)))
    y_pred = y_pred + signal * 0.5
    return t, y_pred


# ---------------------------------------------------------
# 5. Graph explanation generator
# ---------------------------------------------------------
def explain_graph(signal, t, label):
    """Generate a simple language description of the waveform."""
    direction = "upward (toward creativity and experimentation)" if label == "Novelty" else "downward (toward tradition and structure)"
    volatility = "steady" if np.std(signal) < 0.3 else "highly dynamic"
    return f"The curve shows a {volatility} movement leaning {direction} — suggesting that society is entering a {label}-dominated phase."


# ---------------------------------------------------------
# 6. Human-readable summary
# ---------------------------------------------------------
def describe_prediction(label, confidence=0.7, data=None, model=None):
    """Summarize what the AI thinks and why."""
    if label == "Novelty":
        tone = "Creativity and experimentation are on the rise."
        implication = "Expect new ideas, unconventional design, and youthful energy."
    else:
        tone = "Cultural focus is turning toward structure and stability."
        implication = "Expect classic aesthetics, tradition revival, and long-term values."

    return (
        f"**Model Interpretation:** {tone}\n\n"
        f"**Cultural Outlook:** {implication}\n\n"
        f"**Confidence:** {confidence:.0%}"
    )
