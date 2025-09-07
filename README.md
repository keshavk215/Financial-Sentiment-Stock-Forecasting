# Financial Sentiment & Stock Forecasting

An end-to-end project that leverages Natural Language Processing (NLP) on financial news headlines to forecast stock market movements. This project demonstrates the ability to work with unstructured text data, combine it with traditional time-series data, and build a predictive model with a backtested strategy.

## Key Features

- **Automated ETL Pipeline**: Ingests over 50,000 news headlines from multiple sources and historical stock price data.
- **Advanced NLP Modeling**: Uses **FinBERT**, a powerful transformer model from the Hugging Face library, to generate nuanced sentiment scores specifically for financial text.
- **Hybrid Time-Series Forecasting**: Develops an LSTM neural network that integrates both historical price data and the derived sentiment scores to make predictions.
- **Rigorous Backtesting**: Implements a backtesting engine to evaluate the trading strategy's performance on historical data, using key metrics like the Sharpe Ratio and Maximum Drawdown.
- **API Deployment**: The final trained model is deployed as a Dockerized Flask API to serve real-time trading signals.

## Tech Stack

- **Language**: Python 3
- **Data Science**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras
- **NLP**: Hugging Face Transformers (for FinBERT)
- **Deployment**: Flask, Docker
- **Development**: Jupyter Notebook

## Project Workflow

1. **Data Ingestion**: A script fetches daily financial news headlines and historical stock price data (OHLCV).
2. **Sentiment Analysis**: The FinBERT model processes each headline to generate a sentiment score (e.g., from -1 for very negative to +1 for very positive). These scores are aggregated daily.
3. **Feature Engineering**: The daily sentiment scores are merged with the stock price data to create a unified time-series dataset. Lag features and moving averages are created.
4. **Model Training**: An LSTM model is trained on this combined dataset to learn patterns and predict the next day's price movement.
5. **Backtesting & Evaluation**: The model's predictions are used to simulate a trading strategy on a hold-out test set of historical data.
6. **Deployment**: The trained model and sentiment analysis pipeline are packaged into a Flask API and containerized with Docker.

## Setup and Installation (Placeholder)

1. Clone the repository: `git clone <your-repo-link>`
2. Navigate to the directory: `cd financial-sentiment-forecasting`
3. Install dependencies: `pip install -r requirements.txt`

## Usage (Placeholder)

1. Run the full data pipeline and model training: `python scripts/run_pipeline.py`
2. Run the prediction API: `cd api && docker-compose up --build`
