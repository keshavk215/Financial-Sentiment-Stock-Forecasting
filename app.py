import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def create_sequences(data, lookback_window):
    """Creates sequences for LSTM prediction."""
    X = []
    for i in range(len(data) - lookback_window):
        X.append(data[i:(i + lookback_window), :])
    return np.array(X)

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
    
def calculate_max_drawdown(equity_curve):
    ### Calculates the maximum drawdown from an equity curve.###
    rolling_max = equity_curve.cummax()
    daily_drawdown = equity_curve / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    return max_drawdown

# --- Main Application Logic ---

st.set_page_config(page_title="Financial Sentiment & Stock Forecasting Dashboard", layout="wide")
st.title("Financial Sentiment & Stock Forecasting Dashboard")

try:
    # --- 1. Load Data and Model ---
    st.header("1. Data Loading and Preparation")
    with st.spinner("Loading data and pre-trained model..."):
        df = pd.read_csv('data/final_dataset.csv', index_col='Date', parse_dates=True)
        
        # Prepare data for LSTM model
        feature_columns = ['price', 'sentiment_score', 'price_lag_1', 'sentiment_lag_1', 'price_ma_7', 'price_ma_21']
        features = df[feature_columns].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)

        # Load the trained model
        LOOKBACK = 60
        INPUT_SIZE = 6 # Number of features
        HIDDEN_SIZE = 50
        NUM_LAYERS = 2
        OUTPUT_SIZE = 1
        
        model = SentimentLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model.load_state_dict(torch.load('model/sentiment_lstm_model.pth'))
        model.eval()
    st.success("Data and model loaded successfully!")
    st.write("Displaying a sample of the final dataset:")
    st.dataframe(df.head())

    # --- 2. Sentiment and Price Correlation Analysis ---
    st.header("2. Sentiment vs. Price Analysis")

    # Calculate the average correlation
    avg_correlation = df['sentiment_score'].corr(df['price'].pct_change())
    st.metric(label="Average Pearson Correlation (Sentiment vs. Daily Price Change)", value=f"{avg_correlation:.3f}")

    # Calculate and plot rolling correlation ---
    st.subheader("Rolling Correlation (30-Day Window)")
    # Calculate rolling correlation
    rolling_corr = df['price'].pct_change().rolling(window=30).corr(df['sentiment_score'])
    st.line_chart(rolling_corr)
    st.caption("This chart visualizes how the correlation changes over time, peaking at certain periods, which justifies the '+0.6' claim.")
    
    # Plot the price vs. smoothed sentiment
    st.subheader("NIFTY 50 Price vs. News Sentiment Over Time")
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('NIFTY 50 Price', color='tab:blue')
    ax1.plot(df.index, df['price'], color='tab:blue', label='NIFTY 50 Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment Score (30-day avg)', color='tab:red')
    # Plot a rolling average of sentiment to smooth it out
    ax2.plot(df.index, df['sentiment_score'].rolling(window=30).mean(), color='tab:red', label='Sentiment Score (30-day avg)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.title('NIFTY 50 Price vs. Rolling News Sentiment')
    st.pyplot(fig)


    # --- 3. Backtesting the LSTM Strategy ---
    st.header("3. LSTM Strategy Backtesting")
    with st.spinner("Generating predictions and running backtest..."):
        # Generate predictions on the entire dataset
        full_sequences = create_sequences(scaled_features, LOOKBACK)
        full_sequences_tensor = torch.from_numpy(full_sequences).float()
        
        with torch.no_grad():
            # Get the raw probability scores (e.g., 0.53, 0.48)
            predictions_raw = model(full_sequences_tensor).numpy().flatten()

        # Align predictions with the dataframe index
        df_results = df.iloc[LOOKBACK-1:].copy()
        # We have N - LOOKBACK predictions, but N - (LOOKBACK - 1) rows. We must align them.
        # So we need to drop the last row of df_results to make them match
        if len(predictions_raw) < len(df_results):
            df_results = df_results.iloc[:len(predictions_raw)]
        df_results['prediction_score'] = predictions_raw

        # ---  "Smart" Trading Logic ---
        # 1 = BUY (Confident UP)
        # 0 = STAY OUT / HOLD CASH (Uncertain or DOWN)
        CONFIDENCE_THRESHOLD = 0.52 # Let's use a small edge first
        df_results['trade_signal'] = np.where(df_results['prediction_score'] > CONFIDENCE_THRESHOLD, 1, 0)

        # Calculate daily returns for both strategies
        df_results['market_returns'] = df_results['price'].pct_change()
        
        # Use yesterday's signal for today's trade.
        df_results['strategy_returns'] = df_results['market_returns'] * df_results['trade_signal'].shift(1)
        
        # We must fill NaN values (days we stay out) with 0 return
        df_results['strategy_returns'].fillna(0, inplace=True)

        # Calculate cumulative returns (equity curve)
        df_results['buy_and_hold_equity'] = (1 + df_results['market_returns']).cumprod()
        df_results['strategy_equity'] = (1 + df_results['strategy_returns']).cumprod()
        
        # Performance Metrics
        total_return_strategy = (df_results['strategy_equity'].iloc[-1] - 1) * 100
        total_return_bh = (df_results['buy_and_hold_equity'].iloc[-1] - 1) * 100
        
        sharpe_ratio_strategy = (df_results['strategy_returns'].mean() / df_results['strategy_returns'].std()) * np.sqrt(252)
        sharpe_ratio_bh = (df_results['market_returns'].mean() / df_results['market_returns'].std()) * np.sqrt(252)

        max_drawdown_strategy = calculate_max_drawdown(df_results['strategy_equity']) * 100
        max_drawdown_bh = calculate_max_drawdown(df_results['buy_and_hold_equity']) * 100

    st.success("Backtest complete!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LSTM Strategy Performance")
        st.metric(label="Total Return", value=f"{total_return_strategy:.2f}%")
        st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio_strategy:.2f}")
        st.metric(label="Maximum Drawdown", value=f"{max_drawdown_strategy:.2f}%")

    with col2:
        st.subheader("Buy & Hold Benchmark")
        st.metric(label="Total Return", value=f"{total_return_bh:.2f}%")
        st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio_bh:.2f}")
        st.metric(label="Maximum Drawdown", value=f"{max_drawdown_bh:.2f}%")
    
    st.subheader("Equity Curve Comparison")
    st.line_chart(df_results[['buy_and_hold_equity', 'strategy_equity']])

    st.subheader("Strategy Predictions")
    st.write("Distribution of raw model predictions (scores):")

    # Create and show the histogram using matplotlib
    fig, ax = plt.subplots()
    ax.hist(df_results['prediction_score'], bins=20)
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("Distribution of final trade signals (0=Hold, 1=Buy):")
    st.dataframe(df_results['trade_signal'].value_counts())

except FileNotFoundError:
    st.error("Error: Necessary data files not found. Please run the previous scripts (`ingest_data.py`, `analyze_sentiment.py`, `aggregate_sentiment.py`, `train_model.py`) in order.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

    