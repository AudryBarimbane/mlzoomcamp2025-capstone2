#!/usr/bin/env python
# coding: utf-8

# In[40]:


import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = Path("models/best_ml_model.pkl")
FEATURES_PATH = Path("models/features.json")
DATA_PATH = Path("data/processed/nasdaq100_ml_dataset.csv")



# In[42]:


# Load trained model
model = joblib.load(MODEL_PATH)

# Load feature list (order is CRITICAL)
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

print("Model and feature list loaded successfully")
print("Features:", feature_names)



# In[44]:


df = pd.read_csv(DATA_PATH)

# Keep last available row (most recent day)
latest_row = df.iloc[-1]

X_input = pd.DataFrame(
    [latest_row[feature_names].values],
    columns=feature_names
)

X_input



# In[46]:


prediction = model.predict(X_input)[0]

prediction



# In[48]:


def decode_regime(value):
    if value == 0:
        return "Bear Market"
    elif value == 1:
        return "Sideways Market"
    elif value == 2:
        return "Bull Market"
    else:
        return "Unknown"

decode_regime(prediction)



# In[50]:


custom_input = {
    "Open": 15000,
    "High": 15200,
    "Low": 14900,
    "Close": 15150,
    "Volume": 3200000000,
    "avg_sentiment": 0.18,
    "sentiment_std": 0.12,
    "news_count": 42,
    "Return": 0.0021,
    "MA20": 14980,
    "MA50": 14820,
    "Volatility": 0.011
}

X_custom = pd.DataFrame([custom_input])[feature_names]

pred = model.predict(X_custom)[0]

decode_regime(pred)



# In[52]:


REGIME_MAP = {
    0: "Bear",
    1: "Neutral",
    2: "Bull"
}

print("Market regime:", REGIME_MAP.get(prediction, "Unknown"))


# In[ ]:




