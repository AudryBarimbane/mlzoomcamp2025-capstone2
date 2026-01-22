# NASDAQ-100 Market Regime Prediction Using Price Indicators and Aggregated News Sentiment
** CAPSTONE 2 Project – MLZoomcamp 2025**

---

## 📌 Overview

This project applies **Machine Learning and Deep Learning** techniques to NASDAQ-100 market data to predict short-term **market regimes** (Bear / Sideways / Bull).

It follows the complete **MLZoomcamp end-to-end pipeline**:

- Pick a problem & dataset  
- Describe how ML helps  
- Prepare data & run EDA  
- Train several models & select the best  
- Export the trained model  
- Package the model as a FastAPI service  
- Deploy the model with Docker  

This repository includes:

- Market and sentiment datasets  
- Jupyter notebooks (EDA, feature engineering, training, prediction)  
- Python scripts (`train.py`, `train_lstm.py`, `predict.py`, `api.py`)  
- Trained model files  
- Dockerfile for deployment  

---

## 🎯 1. Problem Definition

Financial markets evolve through different **market regimes** influenced by price dynamics, volatility, and investor sentiment.

The objective of this project is to build a machine learning system that classifies the **next-day market regime** of the NASDAQ-100 index into:

- **Bear market**
- **Sideways market**
- **Bull market**

This system can be used for:

- Market regime monitoring  
- Risk management  
- Strategy adaptation  
- Educational and analytical purposes  

---

## 📚 2. Dataset

Two data sources are used:

### 📈 Market Prices
- File: `nasdaq100_prices.csv`
- Frequency: Daily
- Source: Historical NASDAQ-100 price data

### 📰 News Sentiment
https://www.kaggle.com/datasets/enaa0o0/nasdaq-news-sentiment
- File: `nasdaq_news_sentiment.csv`
- Aggregated daily news sentiment indicators:
  - Average sentiment
  - Sentiment standard deviation
  - News count

### 📊 Final engineered dataset

Stored in:  data/processed/nasdaq100_ml_dataset.csv


Columns include:

date,
Open, High, Low, Close, Volume,
avg_sentiment, sentiment_std, news_count,
Return, MA20, MA50, Volatility,
Target



Target encoding:
- `0` → Bear  
- `1` → Sideways  
- `2` → Bull  

---

## 🔎 3. Exploratory Data Analysis (EDA)

EDA is performed in `nasdaq_analysis.ipynb` and includes:

- Market data inspection  
- Missing value analysis  
- Feature distribution analysis  
- Correlation analysis  
- Market regime distribution  
- Relationship between sentiment and returns  
- Volatility behavior across regimes  

This step ensures data quality and validates feature relevance before modeling.

---

## 🧠 4. Model Training

### Data split (time-based)
A **chronological split** is used to avoid data leakage:

- **60%** Train  
- **20%** Validation  
- **20%** Test  

---

### Models evaluated

#### Machine Learning (Baselines)
- Logistic Regression  
- Random Forest  
- XGBoost  

#### Deep Learning
- LSTM (Long Short-Term Memory) network  
  - Sequence length (lookback): 30 days  

---

### Evaluation metrics
- Accuracy  
- F1-macro score  

---

### Results (Test set)

| Model               | F1-macro |
|--------------------|----------|
| Logistic Regression | **0.3576** |
| Random Forest       | 0.1628 |
| XGBoost             | 0.2197 |
| LSTM                | 0.2528 |

📌 **Logistic Regression achieved the best overall performance** and is selected as the final model for deployment.

---

### Training workflow available in
- `train.ipynb`
- `train_lstm.ipynb`
- `train.py`
- `train_lstm.py`

Target encoding:
- `0` → Bear  
- `1` → Sideways  
- `2` → Bull  

---

## 🔎 3. Exploratory Data Analysis (EDA)

EDA is performed in `nasdaq_analysis.ipynb` and includes:

- Market data inspection  
- Missing value analysis  
- Feature distribution analysis  
- Correlation analysis  
- Market regime distribution  
- Relationship between sentiment and returns  
- Volatility behavior across regimes  

This step ensures data quality and validates feature relevance before modeling.

---

## 🧠 4. Model Training

### Data split (time-based)
A **chronological split** is used to avoid data leakage:

- **60%** Train  
- **20%** Validation  
- **20%** Test  

---

### Models evaluated

#### Machine Learning (Baselines)
- Logistic Regression  
- Random Forest  
- XGBoost  

#### Deep Learning
- LSTM (Long Short-Term Memory) network  
  - Sequence length (lookback): 30 days  

---

### Evaluation metrics
- Accuracy  
- F1-macro score  

---

### Results (Test set)

| Model               | F1-macro |
|--------------------|----------|
| Logistic Regression | **0.3576** |
| Random Forest       | 0.1628 |
| XGBoost             | 0.2197 |
| LSTM                | 0.2528 |

📌 **Logistic Regression achieved the best overall performance** and is selected as the final model for deployment.

---

### Training workflow available in
- `train.ipynb`
- `train_lstm.ipynb`
- `train.py`
- `train_lstm.py`

---
Exported files:

models/best_ml_model.pkl <br>
models/scaler.pkl <br>
models/features.json


---

## 🔮 5. Prediction Pipeline

Available in:
- `predict.ipynb`
- `predict.py`

Prediction workflow:

- Load trained model  
- Load scaler and feature order  
- Validate and preprocess input  
- Generate market regime prediction  

---

## 🚀 6. FastAPI Web Service

The file `api.py` provides a real-time prediction API.

### Endpoints

**GET /**
- Health check

**POST /predict**
- Accepts JSON input and returns the predicted market regime

### Example input
```json
{
  "Open": 15000,
  "High": 15120,
  "Low": 14900,
  "Close": 15080,
  "Volume": 3500000000,
  "avg_sentiment": 0.12,
  "sentiment_std": 0.35,
  "news_count": 42,
  "Return": 0.003,
  "MA20": 14950,
  "MA50": 14780,
  "Volatility": 0.011
}

```

### Swagger UI

👉 http://localhost:8000/docs

## 🐳 7. Docker Deployment

This project is fully containerized.

### Step 1 — Build the Docker Image

docker build -t nasdaq-regime-api .


### Step 2 — Run the Container

docker run -d -p 8000:8000 nasdaq-regime-api

### Step 3 — Access the API

👉 http://localhost:8000/docs

## 📂 Repository Structure


MLzoomcamp_Capstone2/
│
├── data/
│   ├── raw/
│   │   ├── nasdaq100_prices.csv
│   │   └── nasdaq_news_sentiment.csv
│   └── processed/
│       └── nasdaq100_ml_dataset.csv
│
├── nasdaq_analysis.ipynb
├── train.ipynb
├── train_lstm.ipynb
├── predict.ipynb
│
├── train.py
├── train_lstm.py
├── predict.py
├── api.py
│
├── models/
│   ├── best_ml_model.pkl
│   ├── scaler.pkl
│   └── features.json
│
├── Dockerfile
├── requirements.txt
└── README.md

## Run Locally (Windows 11 + WSL Recommended)

All commands should be run inside Ubuntu (WSL).

### 1️⃣ Clone the repository

- cd ~ 

- git clone https://github.com/AudryBarimbane/mlzoomcamp2025-capstone2.git

- cd mlzoomcamp2025-capstone2

### 2️⃣ Create virtual environment

- python3 -m venv venv
- source venv/bin/activate


### 3️⃣ Install dependencies

- pip install -r requirements.txt


### 4️⃣ Start FastAPI service

- python api.py


API available at:
👉 http://127.0.0.1:8000/docs

## 🐳 Run Using Docker (Recommended)
✔ Build the Docker image
- docker build -t nasdaq-regime-api .
  
✔ Run the container
- docker run -d -p 8000:8000 nasdaq-regime-api

👉 http://localhost:8000/docs


## ✅ Key Skills Demonstrated

✔ Financial time-series analysis
✔ Feature engineering
✔ Machine Learning & LSTM modeling
✔ Proper time-based data splitting
✔ Model evaluation and selection
✔ FastAPI deployment
✔ Docker containerization

