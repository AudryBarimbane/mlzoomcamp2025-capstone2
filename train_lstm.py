#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().system('pip install tensorflow')



# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[5]:


# =====================
# CONFIG
# =====================
DATA_PATH = Path("data/processed/nasdaq100_ml_dataset.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

LOOKBACK = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =====================
# DATA LOADING
# =====================
def load_data():
    df = pd.read_csv(DATA_PATH)

    y = df["Target"].values
    X = df.drop(columns=["date", "Target"]).values

    return X, y


def temporal_train_val_test_split(X, y, train_size=0.6, val_size=0.2):
    n = len(X)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, scaler

def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []

    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

def main():
    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(X, y)

    X_train_s, X_val_s, X_test_s, scaler = scale_data(
        X_train, X_val, X_test
    )

    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, LOOKBACK)
    X_val_seq, y_val_seq = create_sequences(X_val_s, y_val, LOOKBACK)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, LOOKBACK)

    model = build_lstm(
        input_shape=(LOOKBACK, X_train_seq.shape[2]),
        num_classes=len(np.unique(y))
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH / "best_lstm_model.keras",
            save_best_only=True
        )
    ]

    model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # =====================
    # TEST EVALUATION
    # =====================
    y_pred = np.argmax(model.predict(X_test_seq), axis=1)

    acc = accuracy_score(y_test_seq, y_pred)
    f1 = f1_score(y_test_seq, y_pred, average="macro")

    print("\nLSTM Results (TEST)")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-macro : {f1:.4f}")

    import joblib
    joblib.dump(scaler, MODEL_PATH / "scaler_lstm.pkl")



def build_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()


# In[ ]:




