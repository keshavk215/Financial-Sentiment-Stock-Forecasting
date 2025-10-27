
# Financial Sentiment & Stock Forecasting

An end-to-end data science project that tests the hypothesis: **"Can the sentiment of financial news predict the direction of the stock market?"**

This pipeline ingests raw news headlines, processes them using a **FinBERT** transformer model, trains a **PyTorch LSTM** model to forecast market direction, and evaluates a trading strategy based on the results. The project is deployed in two ways:

1. An interactive **Streamlit Dashboard** for data visualization and backtesting.
2. A production-ready **Dockerized Flask API** to serve real-time predictions.

## Key Findings & Features

* **Statistical Edge:** The final LSTM model achieved a **53% accuracy** on the unseen test set, proving a small but significant predictive edge over a 50% random baseline.
* **NLP with Transformers:** Utilizes the `ProsusAI/finbert` model from Hugging Face for accurate, domain-specific sentiment analysis of 50,000+ news headlines.
* **Feature Engineering:** Enriches the dataset with lagging variables and moving averages (`price_lag_1`, `price_ma_7`, etc.) to improve model performance.
* **Quantitative Backtesting:** The final "smart" trading strategy (which only trades on high-confidence signals) achieved a  **Sharpe Ratio of 0.95** , successfully outperforming the "Buy and Hold" benchmark's Sharpe of 0.76.
* **Risk Management:** The model-driven strategy demonstrated superior risk management, cutting the  **Maximum Drawdown from -15.77% (Benchmark) to just -4.73%** .
* **Dual Deployment:** The project includes both an analytical dashboard (`streamlit run app.py`) and a production-ready, containerized API (`docker run sentiment-api`).

## Tech Stack

* **Language:** Python
* **Data Science & ML:** Pandas, NumPy, Scikit-learn (for scaling), **PyTorch** (for LSTM)
* **NLP:** Hugging Face `transformers` (FinBERT)
* **Data Ingestion:** `yfinance` (for prices), `gnews` (for news)
* **Deployment (API):** Flask, Docker, Gunicorn
* **Deployment (Dashboard):** Streamlit, Matplotlib

## Project Workflow

This project is broken into a series of scripts that must be run in order.

1. **`data_ingestion.py`** : The "Extract" step. Fetches raw news headlines from `gnews` and NIFTY 50 price data from `yfinance`.
2. **`analyze_sentiment.py`** : The first "Transform" step. Loads the raw headlines, runs each one through the FinBERT model, and saves the individual sentiment scores.
3. **`aggregate_sentiment.py`** : The second "Transform" step. Aggregates daily sentiment scores and engineers new features (lags, moving averages), merging them with price data to create the master `final_dataset.csv`.
4. **`train_model.py`** : The "Model" step. Trains the PyTorch LSTM on the `final_dataset.csv` and saves the final, trained model as `sentiment_lstm_model.pth`.
5. **`app.py`** : The "Analytics" deployment. Runs the Streamlit dashboard to visualize results and perform the backtest.
6. **`api.py` & `Dockerfile`** : The "Production" deployment. Defines the Flask API and the Docker container to serve the trained model in real time.

## Setup and Installation

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/keshavk215/Financial-Sentiment-Stock-Forecasting
   ```
2. Create and activate a Python virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install all required dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Ensure you have **Docker Desktop** installed and running on your system to use the API.

## Usage (How to Run)

There are three ways to run this project. You must run the pipeline first.

### 1. Run the Full Data Pipeline (One-Time Setup)

Run these scripts in order from your terminal to generate the data and train the model.

```
python data_ingestion.py
python analyze_sentiment.py
python aggregate_sentiment.py
python train_model.py
```

### 2. Run the Interactive Streamlit Dashboard

To explore the data, see the correlation charts, and view the backtest results.

```
streamlit run app.py
```

### 3. Run the Production-Ready Docker API

To build and run the Docker container that serves your model as a real API.

```
# 1. Build the Docker image
docker build -t sentiment-api .

# 2. Run the container in the background
docker run -d -p 5000:5000 sentiment-api

# 3. Test the API in your browser or Postman
# Go to: http://localhost:5000
```
