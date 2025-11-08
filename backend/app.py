from fastapi import FastAPI

app = FastAPI(title="Pattern Wheel")

@app.get("/")
def root():
    return {"ok": True, "msg": "Pattern Wheel API is alive"}
