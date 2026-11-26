"""FastAPI backend for the Bus Crowd & Delay Predictor.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
Train the models first with:  python train.py
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

FEATURES = ["route_id", "hour", "weekday", "rain"]
ROUTES = {1: "12 Campus", 2: "7 Market Loop", 3: "25 Highway Express"}

app = FastAPI(title="Bus Crowd & Delay Predictor", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

delay_model = crowd_model = None
if os.path.exists("models/delay_model.joblib") and os.path.exists("models/crowd_model.joblib"):
    delay_model = joblib.load("models/delay_model.joblib")
    crowd_model = joblib.load("models/crowd_model.joblib")


class TripQuery(BaseModel):
    route_id: int = Field(..., ge=1, le=3, description="1 = Campus, 2 = Market Loop, 3 = Highway Express")
    hour: int = Field(..., ge=0, le=23)
    weekday: int = Field(..., ge=0, le=6, description="0 = Monday")
    rain: int = Field(0, ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": delay_model is not None}


@app.post("/predict")
def predict(q: TripQuery):
    if delay_model is None or crowd_model is None:
        raise HTTPException(status_code=503, detail="Models not found. Run 'python train.py' first.")
    row = pd.DataFrame([[q.route_id, q.hour, q.weekday, q.rain]], columns=FEATURES)
    delay = float(delay_model.predict(row)[0])
    probs = dict(zip(crowd_model.classes_, crowd_model.predict_proba(row)[0]))
    level = max(probs, key=probs.get)
    return {
        "route": ROUTES[q.route_id],
        "predicted_delay_min": round(max(delay, 0), 1),
        "crowd_level": level,
        "crowd_confidence": round(float(probs[level]), 2),
        "note": "Advisory only. Trained on synthetic data in this prototype.",
    }
