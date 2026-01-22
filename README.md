# NASDAQ-100 Market Regime Prediction  
**Midterm CAPSTONE Project – MLZoomcamp 2025**

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
- File: `nasdaq_news_sentiment.csv`
- Aggregated daily news sentiment indicators:
  - Average sentiment
  - Sentiment standard deviation
  - News count

### 📊 Final engineered dataset

Stored in:
