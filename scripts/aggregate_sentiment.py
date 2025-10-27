import pandas as pd

def create_daily_sentiment_score(headlines_df):
    """
    Aggregates individual headline sentiments into a single daily score.

    Args:
        headlines_df (pd.DataFrame): DataFrame with sentiment for each headline.

    Returns:
        pd.DataFrame: A DataFrame with a single aggregated sentiment score per day.
    """
    print("Aggregating sentiment scores...")
    
    # Ensure the date column is in the correct format
    headlines_df['published_date'] = pd.to_datetime(headlines_df['published_date'])
    
    # Map sentiment labels to numerical values
    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    headlines_df['sentiment_numeric'] = headlines_df['sentiment_label'].map(sentiment_map)
    
    # Group by date and calculate the mean sentiment for each day
    daily_sentiment = headlines_df.groupby('published_date')['sentiment_numeric'].mean()
    
    # Convert to a DataFrame
    daily_sentiment_df = daily_sentiment.to_frame(name='sentiment_score')
    
    print(f"Created daily sentiment scores for {len(daily_sentiment_df)} days.")
    return daily_sentiment_df

def merge_and_engineer_features(sentiment_df, price_df):
    """
    Merges daily sentiment scores with daily stock prices.

    Args:
        sentiment_df (pd.DataFrame): DataFrame with daily sentiment scores.
        price_df (pd.DataFrame): DataFrame with daily stock prices.

    Returns:
        pd.DataFrame: A merged DataFrame ready for modeling.
    """
    print("Merging sentiment scores with price data...")
    
    # Ensure both DataFrames have a datetime index
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    price_df.index = pd.to_datetime(price_df.index)
    
    # Merge the two dataframes on their date index
    # We use a 'left' join to keep all the price data points
    merged_df = price_df.join(sentiment_df, how='left')
    
    # The news cycle might lag. Let's forward-fill the sentiment score for weekends/holidays
    # This assumes the sentiment from Friday carries over Saturday and Sunday
    merged_df['sentiment_score'].fillna(method='ffill', inplace=True)

    # Feature Engineering
    print("Engineering new features (lags and moving averages)...")
    
    # 1. Lagging Variables
    # We lag both price and sentiment. This gives the model info on yesterday's values.
    merged_df['price_lag_1'] = merged_df['price'].shift(1)
    merged_df['sentiment_lag_1'] = merged_df['sentiment_score'].shift(1)

    # 2. Moving Averages
    # We use a 7-day and 21-day moving average to capture short and medium-term trends.
    merged_df['price_ma_7'] = merged_df['price'].rolling(window=7).mean()
    merged_df['price_ma_21'] = merged_df['price'].rolling(window=21).mean()
    
    # Drop all NaN values created by the lags and rolling windows
    merged_df.dropna(inplace=True)
    
    print(f"Final merged and engineered dataset has {len(merged_df)} rows.")
    return merged_df

if __name__ == '__main__':
    try:
        # Load the data
        headlines_with_sentiment = pd.read_csv('data/headlines_with_sentiment.csv')
        nifty_prices = pd.read_csv('data/nifty50_prices.csv', index_col='Date', parse_dates=True)
        
        # --- Create Daily Sentiment Score ---
        daily_sentiment = create_daily_sentiment_score(headlines_with_sentiment)
        
        # --- merge and feature engineering function ---
        final_dataset = merge_and_engineer_features(daily_sentiment, nifty_prices)
        
        # Save the final dataset
        final_dataset.to_csv('data/final_dataset.csv')
        
        print("\nFinal model-ready dataset saved to final_dataset.csv")
        print("Sample of the final dataset:")
        print(final_dataset.head())

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you have run the previous scripts successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
