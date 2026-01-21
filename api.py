#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel


# In[7]:


# =====================
# PATHS
# =====================
MODEL_PATH = Path("models/best_ml_model.pkl")
FEATURES_PATH = Path("models/features.json")

# =====================
# LOAD MODEL & FEATURES
# =====================
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

# =====================
# FASTAPI APP
# =====================
app = FastAPI(title="NASDAQ100 Market Regime API")

# =====================
# INPUT SCHEMA
# =====================
class MarketFeatures(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    avg_sentiment: float
    sentiment_std: float
    news_count: float
    Return: float
    MA20: float
    MA50: float
    Volatility: float

# =====================
# UTILS
# =====================
def decode_regime(value: int) -> str:
    if value == 0:
        return "Bear Market"
    elif value == 1:
        return "Sideways Market"
    elif value == 2:
        return "Bull Market"
    else:
        return "Unknown"

# =====================
# ENDPOINTS
# =====================
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: MarketFeatures):
    data = pd.DataFrame([features.dict()])

    # Ensure correct column order
    data = data[feature_names]

    prediction = model.predict(data)[0]

    return {
        "prediction_class": int(prediction),
        "market_regime": decode_regime(prediction)
    }


# In[7]:





# In[ ]:




