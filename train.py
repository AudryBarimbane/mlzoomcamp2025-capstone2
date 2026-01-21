#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


# In[6]:


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



def temporal_train_val_test_split(X, y, train_size=0.6, val_size=0.2):
    n = len(X)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test




def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    return val_acc, val_f1, test_acc, test_f1

def main():
    X, y = load_data()
    feature_names = list(X.columns)


    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(X, y)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                multi_class="auto",
                random_state=RANDOM_STATE
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE
        )
    }

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

    print("\nTraining ML models (60/20/20 split)...\n")

    for name, model in models.items():
        val_acc, val_f1, test_acc, test_f1 = train_and_evaluate(
            model,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        )

        results[name] = {
            "val_f1": val_f1,
            "test_f1": test_f1
        }

        print(
            f"{name} | "
            f"Val F1: {val_f1:.4f} | "
            f"Test F1: {test_f1:.4f}"
        )

    best_model_name = max(results, key=lambda k: results[k]["val_f1"])
    best_model = models[best_model_name]

    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    joblib.dump(best_model, MODEL_PATH / "best_ml_model.pkl")

    # save feature order
    

    import json
    with open(MODEL_PATH / "features.json", "w") as f:
        json.dump(feature_names, f)

    print("Best model selected:", best_model_name)
    print("Saved to models/best_ml_model.pkl")
    print("Feature list saved to models/features.json")

    


if __name__ == "__main__":
    main()



# In[ ]:




