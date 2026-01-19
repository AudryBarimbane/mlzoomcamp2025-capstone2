# NASDAQ-100 Market Regime Detection Using Price Indicators and News Sentiment Features
Midterm CAPSTONE Project – MLZoomcamp 2025

---

## 📌 Overview

This project applies **Machine Learning and Deep Learning** techniques to financial market data from the **NASDAQ-100 index** to detect daily **market regimes** (*Bear*, *Sideways*, *Bull*).

The project follows the complete **MLZoomcamp end-to-end pipeline**:

- Pick a problem & dataset  
- Describe how ML helps  
- Prepare data & perform Exploratory Data Analysis (EDA)  
- Train multiple models and select the best one  
- Export the trained model  
- Build a prediction pipeline  
- Serve predictions using FastAPI  
- Deploy the application with Docker  

This repository includes:

- Market and sentiment datasets  
- Jupyter notebooks (EDA, ML training, LSTM experiments, prediction)  
- Python scripts (`train.py`, `predict.py`, `api.py`)  
- Trained model files  
- Dockerfile for deployment  

---

## 🎯 1. Problem Definition

Financial markets move through different **regimes** depending on price dynamics, volatility, and external information such as financial news.

The goal of this project is to build a **multi-class classification model** that predicts the **next-day market regime** of the NASDAQ-100 index:

- **Bear**
- **Sideways**
- **Bull**

Potential applications include:

- Market regime monitoring  
- Risk management  
- Trading strategy adaptation  
- Financial decision-support systems  

---

## 📚 2. Dataset

### Market Data
Daily historical price data for the NASDAQ-100 index.

### News Sentiment Data
Aggregated sentiment indicators derived from financial news sources.  
Sentiment features are **pre-computed numerical values** (no NLP model is trained in this project).

### Dataset columns

date,
Open, High, Low, Close, Volume,
avg_sentiment, sentiment_std, news_count,
Return, MA20, MA50, Volatility,
Target



All features are numerical except `date`.

Dataset location:

data/processed/nasdaq100_ml_dataset.csv


---

## 🔎 3. Exploratory Data Analysis (EDA)

EDA is performed in **`nasdaq_analysis.ipynb`** and includes:

- Market data inspection  
- Missing value analysis  
- Feature distribution analysis  
- Correlation heatmaps  
- Market regime distribution  
- Relationship between sentiment and price-based indicators  

This step ensures data quality and model reliability.

---

## 🧠 4. Model Training

### Models evaluated

**Classical Machine Learning**
- Logistic Regression (**best performing model**)  
- Random Forest  
- XGBoost  

**Deep Learning (Experimental)**
- LSTM (Long Short-Term Memory)

### Evaluation metrics
- Accuracy  
- F1-score (macro)

Logistic Regression outperformed LSTM on this structured tabular dataset and was selected as the final model.

### Training files
- `train.ipynb`
- `train.py`

LSTM experiments:
- `train_lstm.ipynb`

### Exported model
models/best_ml_model.pkl


---

## 🔮 5. Prediction Pipeline

Available in:
- `predict.ipynb`
- `predict.py`

Prediction workflow:

- Load trained model  
- Validate feature order  
- Preprocess and scale inputs  
- Generate market regime prediction  

Model output:
- **Bear**
- **Sideways**
- **Bull**

---

## 🚀 6. FastAPI Web Service

The file **`api.py`** provides a real-time prediction API.

### Endpoints

#### `GET /`
Health check endpoint.

#### `POST /predict`
Accepts JSON input and returns the predicted market regime.

Example input:

```json
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

Swagger UI:

http://localhost:8000/docs


7. Docker Deployment

This project is fully containerized.

Build the Docker image
docker build -t nasdaq-regime-api .

Run the container
docker run -d -p 8000:8000 nasdaq-regime-api

Access the API
http://localhost:8000/docs

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

🧪 Run Locally (Without Docker)

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Start the API:

python api.py


API available at:

http://127.0.0.1:8000/docs

🐳 Run Using Docker (Recommended)
docker build -t nasdaq-regime-api .
docker run -d -p 8000:8000 nasdaq-regime-api

✅ Key Skills Demonstrated

Financial time-series analysis

Feature engineering

Machine Learning model comparison

Deep Learning experimentation (LSTM)

FastAPI model serving

Docker containerization

🔮 Limitations & Future Work

Sentiment features are pre-computed (no FinBERT used in this project)

Real-time NLP sentiment analysis could be added

More advanced temporal models could be explored
