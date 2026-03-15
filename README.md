# Stock Price Prediction using Stacked LSTM: An End-to-End Deep Learning Pipeline

This repository hosts a "Major Project" focused on time-series forecasting in the financial domain. Utilizing a **Stacked Long Short-Term Memory (LSTM)** architecture, the project implements a modular, research-oriented pipeline to predict stock closing prices using historical data and technical indicators.

## 🚀 Project Goal
To build a reproducible and scalable deep learning workflow that bridges the gap between raw financial data and actionable insights, culminating in a research paper and a real-time Streamlit dashboard.

---

## 📂 Repository Structure

The project follows a **Layered Design Pattern** to ensure modularity and research integrity.

```text
Stock_Prediction_Project/
├── research/              # Academic workspace (Literature review & Paper drafts)
├── experiments/           # Evidence Vault (Training logs & performance plots)
├── src/                   # The Engine (Modular Python layers)
│   ├── data_loader.py     # Layer 1: Data Ingestion (yfinance API)
│   ├── processor.py       # Layer 2: Feature Engineering & Windowing
│   ├── model_builder.py   # Layer 3: Stacked LSTM Architecture
│   ├── trainer.py         # Layer 4: Training & Validation Logic
│   └── evaluator.py       # Layer 5: Research Metrics (RMSE, MAPE, DA)
├── data/                  # Local data storage (Git-ignored)
├── models/                # Trained model vault (.h5 / .keras files)
├── notebooks/             # Experimental "Scratchpad" for EDA
├── app.py                 # Streamlit Dashboard (Live Demo)
├── main.py                # Pipeline Orchestrator
├── requirements.txt       # Project dependencies
└── .gitignore             # Git exclusion rules
