import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_sequences(data, lookback_window):
    """
    Creates sequences of data for LSTM training.
    
    Args:
        data (np.array): The scaled dataset.
        lookback_window (int): The number of previous time steps to use as input variables.
        
    Returns:
        np.array, np.array: The input sequences (X) and their corresponding labels (y).
    """
    X, y = [], []
    for i in range(len(data) - lookback_window):
        X.append(data[i:(i + lookback_window), :-1]) # All features except the last column (target)
        y.append(data[i + lookback_window - 1, -1])     # The target column
    return np.array(X), np.array(y)

# Define the LSTM Model architecture
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

if __name__ == '__main__':
    try:
        # --- 1. Load and Prepare Data ---
        print("Loading and preparing data...")
        df = pd.read_csv('data/final_dataset.csv', index_col='Date', parse_dates=True)
        
        # Create target variable: 1 if next day's price is up, 0 if down
        df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        
        # Drop the last row as it will have a NaN target
        df.dropna(inplace=True)
        
        # Select features and target
        feature_columns = ['price', 'sentiment_score', 'price_lag_1', 'sentiment_lag_1', 'price_ma_7', 'price_ma_21', 'target']
        features = df[feature_columns].values
        
        # Scale the features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # --- 2. Create Sequences ---
        LOOKBACK = 60
        print(f"Creating sequences with a lookback window of {LOOKBACK} days...")
        X, y = create_sequences(scaled_features, LOOKBACK)
        
        # --- 3. Train/Test Split (Temporal) ---
        # IMPORTANT: For time-series, we do not shuffle the data.
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float().view(-1, 1)
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float().view(-1, 1)

        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # --- 4. Build and Train the LSTM Model ---
        INPUT_SIZE = X_train.shape[2] # Number of features 
        HIDDEN_SIZE = 50
        NUM_LAYERS = 2
        OUTPUT_SIZE = 1
        NUM_EPOCHS = 20
        LEARNING_RATE = 0.001

        model = SentimentLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        criterion = nn.BCELoss() # Binary Cross-Entropy Loss for classification
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("\nStarting model training...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
        
        # --- 5. Evaluate the Model ---
        print("\nEvaluating model on the test set...")
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            print(f'Test Accuracy: {accuracy.item()*100:.2f}%')
            
        # --- 6. Save the Model ---
        torch.save(model.state_dict(), 'model/sentiment_lstm_model.pth')
        print("\nTrained model saved to sentiment_lstm_model.pth")

    except FileNotFoundError:
        print("Error: 'final_dataset.csv' not found. Please run the previous scripts first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
