NASDAQ-100 Market Regime Detection Using Price Indicators and News Sentiment Features

Midterm CAPSTONE Project – MLZoomcamp 2025

📌 Overview

This project applies Machine Learning and Deep Learning techniques to financial market data from the NASDAQ-100 index to detect daily market regimes (Bear, Sideways, Bull).

The project follows the complete MLZoomcamp end-to-end pipeline:

Pick a problem & dataset

Describe how Machine Learning helps

Prepare data & perform Exploratory Data Analysis (EDA)

Train multiple models and select the best one

Export the trained model

Build a prediction pipeline

Package the model as a FastAPI service

Deploy the service using Docker

This repository includes:

Processed market and sentiment datasets

Jupyter notebooks (EDA, training, LSTM experiments, prediction)

Python scripts (train.py, predict.py, api.py)

Trained model file

Dockerfile for deployment

🎯 1. Problem Definition

Financial markets evolve through different market regimes, influenced by price movements, volatility, and external information such as financial news.

The objective of this project is to build a market regime classification system for the NASDAQ-100 index, predicting the next-day regime as:

Bear market

Sideways market

Bull market

This system can be used for:

Market regime monitoring

Risk management

Strategy adaptation

Decision-support tools in finance

📚 2. Dataset
📈 Market Data

Source: NASDAQ-100 historical price data

Frequency: Daily

📰 News Sentiment Data

Aggregated financial news sentiment features

Sentiment scores are pre-computed and provided as numerical inputs

📊 Final dataset columns
date,
Open, High, Low, Close, Volume,
avg_sentiment, sentiment_std, news_count,
Return, MA20, MA50, Volatility,
Target

All features are numerical except date.

📁 Stored in:

data/processed/nasdaq100_ml_dataset.csv
🔎 3. Exploratory Data Analysis (EDA)

EDA is performed in nasdaq_analysis.ipynb and includes:

Market data inspection

Missing value analysis

Feature distribution analysis

Correlation analysis

Market regime distribution

Relationship between sentiment and returns

Volatility behavior across market regimes

EDA confirms that volatility and price-based indicators are strong discriminative features, while sentiment features provide complementary contextual information.

🧠 4. Model Training
Models evaluated

Classical Machine Learning (Baselines):

Logistic Regression (best performance)

Random Forest

XGBoost

Deep Learning (Comparative):

LSTM (Long Short-Term Memory)

Evaluation metrics

Accuracy

F1-score (macro)

F1-macro is used to account for class imbalance.

Training workflow available in:

train.ipynb

train.py

Deep learning experiments are conducted in:

train_lstm.ipynb

Exported file
models/best_ml_model.pkl

The Logistic Regression model is selected as the final model due to superior performance and robustness on structured tabular data.

🔮 5. Prediction Pipeline

Available in:

predict.ipynb

predict.py

Prediction workflow

Load the trained model

Validate feature order

Preprocess input data

Generate market regime prediction

Predictions return one of:

Bear

Sideways

Bull

🚀 6. FastAPI Web Service

The file api.py provides a real-time prediction API.

Endpoints
GET /

Health check endpoint.

POST /predict

Accepts JSON input and returns the predicted market regime.

Example input
{
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
Swagger UI

👉 http://localhost:8000/docs

🐳 7. Docker Deployment

This project is fully containerized.

Step 1 — Build the Docker image
docker build -t nasdaq-regime-api .
Step 2 — Run the container
docker run -d -p 8000:8000 nasdaq-regime-api
Step 3 — Access the API

👉 http://localhost:8000/docs

📂 Repository Structure
nasdaq-market-regime/
│
├── data/
│   └── processed/
│       └── nasdaq100_ml_dataset.csv
│
├── nasdaq_analysis.ipynb
├── train.ipynb
├── train_lstm.ipynb
├── predict.ipynb
│
├── train.py
├── predict.py
├── api.py
│
├── models/
│   └── best_ml_model.pkl
│
├── Dockerfile
├── requirements.txt
└── README.md
