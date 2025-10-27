import yfinance as yf
from gnews import GNews
import pandas as pd
from datetime import date, timedelta

def fetch_news_data(start_date, end_date, query="Indian stock market business economy"):
    """
    Fetches news headlines from Google News for a given date range and query.

    Args:
        start_date (date): The start date for fetching news.
        end_date (date): The end date for fetching news.
        query (str): The search query for Google News.

    Returns:
        pd.DataFrame: A DataFrame containing news headlines and their publication dates.
    """
    print(f"Fetching news for query: '{query}'...")
    google_news = GNews(language='en', country='IN', start_date=start_date, end_date=end_date)
    
    # Use a broader query to get more results
    news_articles = google_news.get_news(query)
    
    if not news_articles:
        print("No news articles found for the given query and date range.")
        return pd.DataFrame(columns=['published_date', 'title'])

    # Convert to a pandas DataFrame
    news_df = pd.DataFrame(news_articles)
    news_df = news_df[['published date', 'title']]
    news_df.rename(columns={'published date': 'published_date'}, inplace=True)

    # Convert 'published_date' to a standard datetime format
    news_df['published_date'] = pd.to_datetime(news_df['published_date']).dt.date
    
    print(f"Successfully fetched {len(news_df)} news articles.")
    return news_df

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock price data for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (date): The start date for the data.
        end_date (date): The end date for the data.

    Returns:
        pd.DataFrame: A DataFrame with the daily adjusted close prices.
    """
    print(f"Fetching stock data for ticker: {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)['Close']
        data = data.dropna()
        print(f"Successfully fetched {len(data)} data points.")
        data.rename(columns={ticker: 'price'}, inplace=True)
        return data
    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")
        return None

if __name__ == '__main__':
    # --- Configuration ---
    # Set the date range for the last 3 years
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=3 * 365)
    
    # Ticker for the NIFTY 50 index
    NIFTY_TICKER = '^NSEI'

    # --- Fetch Data ---
    # Fetch news headlines
    headlines_df = fetch_news_data(START_DATE, END_DATE)
    if not headlines_df.empty:
        headlines_df.to_csv('data/raw_headlines.csv', index=False)
        print("News headlines saved to raw_headlines.csv")
        print(headlines_df.head())

    # Fetch NIFTY 50 price data
    nifty_df = fetch_stock_data(NIFTY_TICKER, START_DATE, END_DATE)
    if nifty_df is not None:
        nifty_df.to_csv('data/nifty50_prices.csv')
        print("\nNIFTY 50 price data saved to nifty50_prices.csv")
        print(nifty_df.head())
