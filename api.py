import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load Model and Helper Functions ---

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SentimentLSTM, self).__init__()
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

# --- 2. Initialize Model and Scaler ---
try:
    # Load the dataset to fit the scaler
    df = pd.read_csv('data/final_dataset.csv', index_col='Date', parse_dates=True)
    feature_columns = ['price', 'sentiment_score', 'price_lag_1', 'sentiment_lag_1', 'price_ma_7', 'price_ma_21']
    
    # Use the SAME scaler that the model was trained on
    SCALER = MinMaxScaler(feature_range=(0, 1))
    SCALER.fit(df[feature_columns].values)
    
    # Load the trained model
    LOOKBACK = 60
    INPUT_SIZE = len(feature_columns)
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    
    MODEL = SentimentLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    MODEL.load_state_dict(torch.load('model/sentiment_lstm_model.pth'))
    MODEL.eval()
    
    print("Model and scaler loaded successfully.")

except FileNotFoundError:
    print("FATAL ERROR: Could not find 'final_dataset.csv' or 'sentiment_lstm_model.pth'.")
    print("Please run the training scripts first.")
    
# --- 3. Create the Flask App ---
app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Sentiment LSTM Model API</h1><p>Send a POST request to /predict</p>"

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the 60-day sequence from the request
        # Expecting JSON like: {"sequence": [[...], [...], ... 60 rows]}
        data = request.json['sequence']
        
        # 1. Convert to numpy array
        sequence_np = np.array(data)
        
        # 2. Check shape
        if sequence_np.shape != (LOOKBACK, INPUT_SIZE):
            return jsonify({"error": f"Invalid sequence shape. Expected ({LOOKBACK}, {INPUT_SIZE}), got {sequence_np.shape}"}), 400
            
        # 3. Scale the data
        sequence_scaled = SCALER.transform(sequence_np)
        
        # 4. Convert to tensor
        sequence_tensor = torch.from_numpy(sequence_scaled).float().unsqueeze(0) # Add batch dimension
        
        # 5. Make prediction
        with torch.no_grad():
            prediction_raw = MODEL(sequence_tensor)
            prediction_score = prediction_raw.item()
        
        # 6. Format response
        direction = "UP" if prediction_score > 0.5 else "DOWN"
        
        return jsonify({
            "prediction": direction,
            "confidence_score": prediction_score
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Setting host='0.0.0.0' makes it accessible on the network, which is needed for Docker
    app.run(debug=True, host='0.0.0.0', port=5000)
