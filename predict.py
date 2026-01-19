#!/usr/bin/env python
# coding: utf-8

# In[2]:


import joblib
import pandas as pd
from pathlib import Path


# In[4]:


MODEL_PATH = Path("models/best_ml_model.pkl")

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "avg_sentiment", "sentiment_std", "news_count",
    "Return", "MA20", "MA50", "Volatility"
]

def predict(input_data: dict):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data])
    df = df[FEATURES]

    prediction = model.predict(df)[0]

    regime_map = {
        0: "Bear",
        1: "Sideways",
        2: "Bull"
    }

    return regime_map[prediction]


if __name__ == "__main__":
    example = {
        "Open": 15000,
        "High": 15200,
        "Low": 14900,
        "Close": 15100,
        "Volume": 3200000000,
        "avg_sentiment": 0.12,
        "sentiment_std": 0.45,
        "news_count": 85,
        "Return": 0.003,
        "MA20": 14980,
        "MA50": 14850,
        "Volatility": 0.014
    }

    print(predict(example))


# In[ ]:




