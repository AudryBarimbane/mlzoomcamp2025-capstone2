#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


# In[3]:


DATA_PATH = Path("data/processed/nasdaq100_ml_dataset.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop date (non numérique)
    df = df.drop(columns=["date"])

    X = df.drop(columns=["Target"])
    y = df["Target"]

    return X, y

def train_and_evaluate(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    acc_scores = []
    f1_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    return np.mean(acc_scores), np.mean(f1_scores)

def main():
    X, y = load_data()

    models = {}

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            random_state=RANDOM_STATE
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE
    )

    if xgb_available:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=3,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss"
        )

    results = {}

    print("\nTraining ML models...\n")

    for name, model in models.items():
        acc, f1 = train_and_evaluate(model, X, y)
        results[name] = {"accuracy": acc, "f1": f1}
        print(f"{name} → Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")

    # Select best model (F1 priority)
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best_model = models[best_model_name]

    # Retrain on full dataset
    best_model.fit(X, y)

    joblib.dump(best_model, MODEL_PATH / "best_ml_model.pkl")

    print("\nBest model selected:", best_model_name)
    print("Saved to models/best_ml_model.pkl")

if __name__ == "__main__":
    main()


# In[ ]:




