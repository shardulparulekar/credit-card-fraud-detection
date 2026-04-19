"""
api/main.py  —  FastAPI Prediction Server
==========================================
Exposes the trained model as a REST API for real-time fraud scoring.

Start:
    uvicorn api.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import os

# ── Load artifacts ─────────────────────────────────────────────────────────────
for f in ["models/best_model.pkl", "models/scaler.pkl", "models/threshold.pkl"]:
    if not os.path.exists(f):
        raise RuntimeError(f"Missing {f}. Run: python pipeline.py")

model      = joblib.load("models/best_model.pkl")
scaler     = joblib.load("models/scaler.pkl")
threshold  = float(joblib.load("models/threshold.pkl"))
model_name = joblib.load("models/model_name.pkl")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

app = FastAPI(
    title="💳 Fraud Detection API",
    description="Real-time ML-powered fraud scoring for payment transactions.",
    version="1.0.0"
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    Time:   float = Field(..., example=80000.0)
    Amount: float = Field(..., example=149.62)
    V1:  float = 0.0;  V2:  float = 0.0;  V3:  float = 0.0;  V4:  float = 0.0
    V5:  float = 0.0;  V6:  float = 0.0;  V7:  float = 0.0;  V8:  float = 0.0
    V9:  float = 0.0;  V10: float = 0.0;  V11: float = 0.0;  V12: float = 0.0
    V13: float = 0.0;  V14: float = 0.0;  V15: float = 0.0;  V16: float = 0.0
    V17: float = 0.0;  V18: float = 0.0;  V19: float = 0.0;  V20: float = 0.0
    V21: float = 0.0;  V22: float = 0.0;  V23: float = 0.0;  V24: float = 0.0
    V25: float = 0.0;  V26: float = 0.0;  V27: float = 0.0;  V28: float = 0.0

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    model_used: str
    threshold_used: float

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model": model_name, "threshold": threshold}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: Transaction):
    try:
        df = pd.DataFrame([txn.dict()])
        df[["Amount_scaled", "Time_scaled"]] = scaler.transform(df[["Amount", "Time"]])
        df = df.drop(columns=["Amount", "Time"])[FEATURE_COLS]

        prob     = float(model.predict_proba(df)[0][1])
        is_fraud = prob >= threshold
        risk     = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.3 else "LOW")

        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            risk_level=risk,
            model_used=model_name,
            threshold_used=round(threshold, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[Transaction]):
    if len(transactions) > 100:
        raise HTTPException(400, "Max 100 transactions per batch.")
    return [predict(t) for t in transactions]
