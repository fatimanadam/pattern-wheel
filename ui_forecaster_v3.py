import streamlit as st
import numpy as np
import plotly.graph_objects as go
from data.model_utils import load_model, predict_axis, explain_graph

# ---------------------------------------------------------
# Setup & Load Model
# ---------------------------------------------------------
st.set_page_config(page_title="The Pendulum Wheel", layout="wide")

@st.cache_resource
def get_model():
    return load_model("backend/axis_model.pkl")

model = get_model()

# ---------------------------------------------------------
# 🌫️ BACKGROUND (Same as Before — Dark, Fluid, X-ray)
# ---------------------------------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  margin: 0 !important;
  padding: 0 !important;
  background: radial-gradient(circle at 50% 50%, rgba(5,5,10,1), rgba(0,0,0,1));
  color: #e8e8f5;
  font-family: 'Inter', sans-serif;
  overflow-x: hidden;
}
[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
}
#xray-bg {
  position: fixed; top: 0; left: 0;
  width: 100vw; height: 100vh;
  z-index: -1; overflow: hidden;
}
#xray-bg::before, #xray-bg::after {
  content: ''; position: absolute; width: 100%; height: 100%;
  filter: blur(140px) brightness(1.7) saturate(160%);
  opacity: 0.85; mix-blend-mode: screen;
}
#xray-bg::before {
  background:
    radial-gradient(1500px at 35% 45%, rgba(255,255,255,0.15), transparent 70%),
    radial-gradient(1000px at 65% 55%, rgba(255,255,255,0.12), transparent 80%);
  animation: move1 14s ease-in-out infinite alternate;
}
#xray-bg::after {
  background:
    radial-gradient(1200px at 50% 50%, rgba(255,255,255,0.1), transparent 75%),
    radial-gradient(800px at 25% 75%, rgba(255,255,255,0.08), transparent 85%);
  animation: move2 18s ease-in-out infinite alternate-reverse;
}
@keyframes move1 {
  0% {transform: translate3d(0,0,0) scale(1);}
  50% {transform: translate3d(-4%,3%,0) scale(1.04);}
  100% {transform: translate3d(3%,-3%,0) scale(1);}
}
@keyframes move2 {
  0% {transform: translate3d(0,0,0) scale(1);}
  50% {transform: translate3d(3%,-3%,0) scale(1.03);}
  100% {transform: translate3d(-3%,3%,0) scale(1);}
}
[data-testid="stChatInput"] textarea {
  background: rgba(25,25,35,0.8);
  border-radius: 20px;
  border: 1px solid rgba(160,160,200,0.25);
  color: #f0f0fa;
  box-shadow: 0 0 10px rgba(255,255,255,0.12);
}
[data-testid="stChatInput"] textarea:focus {
  outline: none !important;
  border: 1px solid rgba(200,200,255,0.45);
}
[data-testid="stPlotlyChart"] {
  background: rgba(15,15,25,0.5);
  border-radius: 20px;
  backdrop-filter: blur(22px);
  box-shadow: 0 0 28px rgba(255,255,255,0.12);
  padding: 12px;
  cursor: grab;
}
[data-testid="stPlotlyChart"]:active { cursor: grabbing; }
</style>
<div id="xray-bg"></div>
<script>
document.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    setTimeout(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 300);
  }
});
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.title("The Pendulum Wheel")
st.write("""
This tool shows how culture swings between two moods — comfort and change.  
Ask questions like *“What will fashion feel like in 2030?”* or *“Why did people love minimal design in 1995?”*
""")

# ---------------------------------------------------------
# Chat Setup
# ---------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------
user_msg = st.chat_input("Ask your question about culture or trends...")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)

    import re
    match = re.search(r"\b(19\\d{2}|20\\d{2}|21\\d{2})\\b", user_msg)
    year = int(match.group(1)) if match else None

    # Detect main topic
    topic = "General Culture"
    if any(k in user_msg.lower() for k in ["makeup", "beauty", "skincare"]):
        topic = "Beauty/Makeup"
    elif any(k in user_msg.lower() for k in ["fashion", "apparel", "clothing"]):
        topic = "Fashion"
    elif any(k in user_msg.lower() for k in ["tech", "technology", "ai", "gadgets"]):
        topic = "Technology"
    elif any(k in user_msg.lower() for k in ["music", "sound", "artist"]):
        topic = "Music"

    novelty_factor = user_msg.lower().count("new") / 3
    order_factor = user_msg.lower().count("classic") / 3
    novelty_factor = np.clip(novelty_factor, 0, 1)
    order_factor = np.clip(order_factor, 0, 1)

    base_features = {
        "gt_search": 0.4 + novelty_factor * 0.4,
        "yt_views": 0.5,
        "sp500_ret": 0.5,
        "cpi_surprise": 0.5 - order_factor * 0.2,
        "unemp_rate": 0.5,
        "youth_proxy": np.sin(0.3 + novelty_factor),
        "shock_signed": np.cos(0.5 + order_factor),
        "novelty_kw_density": novelty_factor,
        "order_kw_density": order_factor,
    }

    label, conf, _ = predict_axis(model, base_features)
    desc = explain_graph(np.sin(np.linspace(0, np.pi, 200)), np.linspace(0, 10, 200), label)

    # --- Real life examples ---
    if label == "Order":
        meaning = (
            "Culture is in an **Order phase** — people want things that feel safe, timeless, and calm. "
            "Think of simple design, natural materials, soft lighting, and trusted brands. "
            "This is when trends slow down, and people value what lasts."
        )
        examples = {
            "Beauty/Makeup": "minimal skin tints, cream textures, glass skin — like **Glossier** or **Ilia**",
            "Fashion": "neutral colors, structured basics — like **COS**, **Everlane**, or **The Row**",
            "Technology": "clean, reliable design — like **Apple** or **Sony’s** simple product lines",
            "Music": "soft vocals, acoustic or nostalgic styles — think **Adele** or **Lana Del Rey**"
        }
    else:
        meaning = (
            "Culture is in a **Novelty phase** — people are drawn to what feels new, risky, or bold. "
            "This is when innovation, youth, and emotion lead. "
            "It’s about color, texture, and breaking old rules."
        )
        examples = {
            "Beauty/Makeup": "chrome eyes, graphic blush, hybrid skincare — like **Rare Beauty** or **Fenty**",
            "Fashion": "experimental silhouettes — like **Rick Owens**, **Diesel**, or **Marine Serre**",
            "Technology": "wearables, AI design, emotional tech — like **Nothing** or **Humane AI Pin**",
            "Music": "hyperpop, experimental sound — think **Charli XCX**, **PinkPantheress**, or **Yeat**"
        }

    tone = examples.get(topic, "independent creators, nostalgic design, and DIY culture are shaping new ideas.")

    reply = (
        f"**You asked:** “{user_msg}”\n\n"
        f"I’m analyzing this through **The Pendulum Wheel** — how culture swings between comfort and change.\n\n"
        f"➡️ For **{topic}**, in **{year or 'this era'}**, the energy leans toward **{'calm and familiar' if label == 'Order' else 'new and expressive'}**.\n\n"
        f"{meaning}\n\n"
        f"**Real-life examples:** {tone}\n\n"
        f"{desc}"
    )

    st.session_state.chat.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.write(reply)

# ---------------------------------------------------------
# Graph Section
# ---------------------------------------------------------
st.markdown("### The Pendulum Wheel: Cultural Shift Over Time")

domains = {
    "Fashion": "#B8A7FF",
    "Technology": "#A3F0E0",
    "Economy": "#FFF5AA",
    "Music": "#DAB3FF",
    "Social Mood": "#A8AFFF"
}
t = np.linspace(0, 20, 200)
fig = go.Figure()
for i, (domain, color) in enumerate(domains.items()):
    signal = np.sin(t / (2 + i * 0.3)) + 0.3 * np.cos(t / (3 + i * 0.2))
    fig.add_trace(go.Scatter3d(
        x=t, y=signal, z=np.full_like(t, i * 0.4),
        mode="lines", line=dict(color=color, width=6), name=domain,
        hovertemplate=f"<b>{domain}</b><br>Year: %{{x:.1f}}<br>Energy: %{{y:.2f}}<extra></extra>"
    ))
fig.update_layout(
    scene=dict(
        xaxis_title="Time (Years)",
        yaxis_title="Cultural Energy (-1 = Calm / +1 = Change)",
        zaxis_title="Domain",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#ddddff"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#ddddff"),
        zaxis=dict(showgrid=False, color="#ddddff"),
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f0f0f8"),
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------
st.markdown("""
### How to Read the Pendulum Wheel

The Pendulum Wheel shows how culture swings between **comfort** and **change**.  
Each colored line shows a different area of life — fashion, music, tech, or mood — and how its “energy” moves over time.

- When lines **rise**, people want *newness*: color, youth, and experimentation.  
- When lines **fall**, people want *stability*: calm, nostalgia, and trust.  
- When lines **cross**, one part of society is changing while another resists.

This helps you see how trends ripple between industries —  
like how a tech shift can inspire fashion or music soon after.
""")

st.caption("Built with the Pendulum Wheel Model — dark neural design, fluid movement, and real-world cultural logic.")
