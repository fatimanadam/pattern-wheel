from fastapi import FastAPI
from pydantic import BaseModel
from data.pattern_model import PatternModel

app = FastAPI(title="Secular Pendulum API", version="1.0.0")

class TrendInput(BaseModel):
    amplitude1: float
    amplitude2: float
    scarcity: float
    youth: float
    coupling: float
    length: int
    shock: int

@app.get("/")
def root():
    return {"message": "Secular Pendulum API is running. Visit /docs for interface."}

@app.post("/score")
def score_trend(input: TrendInput):
    model = PatternModel(
        amplitude1=input.amplitude1,
        amplitude2=input.amplitude2,
        scarcity=input.scarcity,
        youth_weight=input.youth,
        coupling=input.coupling
    )
    t, signal = model.generate_trends(length=input.length, shock_month=input.shock)
    return {"time": t.tolist(), "signal": signal.tolist()}

