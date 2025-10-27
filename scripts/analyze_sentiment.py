import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def analyze_headline_sentiment(headlines_df):
    """
    Analyzes the sentiment of each headline using a pre-trained FinBERT model.

    Args:
        headlines_df (pd.DataFrame): DataFrame containing news headlines.

    Returns:
        pd.DataFrame: The original DataFrame with added 'sentiment_label' and 'sentiment_score' columns.
    """
    print("Loading FinBERT sentiment analysis model... (This may take a moment)")
    # Using a specific FinBERT model fine-tuned for financial sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    results = []
    
    print(f"Analyzing sentiment for {len(headlines_df)} headlines... (This will take a while)")
    # Use tqdm for a progress bar
    for headline in tqdm(headlines_df['title']):
        try:
            # The pipeline returns a list with a dictionary inside
            result = sentiment_pipeline(headline)
            results.append(result[0])
        except Exception as e:
            print(f"Could not process headline: '{headline}'. Error: {e}")
            # Append a neutral result in case of an error
            results.append({'label': 'neutral', 'score': 0.5})

    # Convert the list of results into a DataFrame
    sentiment_df = pd.DataFrame(results)
    sentiment_df.rename(columns={'label': 'sentiment_label', 'score': 'sentiment_score'}, inplace=True)
    
    # Concatenate the original DataFrame with the new sentiment data
    enriched_df = pd.concat([headlines_df.reset_index(drop=True), sentiment_df], axis=1)
    
    print("\nSentiment analysis complete.")
    return enriched_df

if __name__ == '__main__':
    # Load the raw headlines
    try:
        headlines_data = pd.read_csv('data/raw_headlines.csv')
        
        # --- Perform Sentiment Analysis ---
        headlines_with_sentiment = analyze_headline_sentiment(headlines_data)
        
        # Save the results to a new CSV file
        headlines_with_sentiment.to_csv('data/headlines_with_sentiment.csv', index=False)
        
        print("\nEnriched data saved to headlines_with_sentiment.csv")
        print("Sample of the results:")
        print(headlines_with_sentiment.head())
        
    except FileNotFoundError:
        print("Error: 'raw_headlines.csv' not found. Please run 'ingest_data.py' first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
