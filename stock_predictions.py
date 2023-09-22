import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import json
from keras.preprocessing.text import Tokenizer

def tokenizer_from_json_string(json_string):
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config', {})

    # Create a new tokenizer with the config parameters
    tokenizer = Tokenizer(**config)
    tokenizer.word_index = json.loads(tokenizer_config.get('word_index', '{}'))
    tokenizer.index_word = json.loads(tokenizer_config.get('index_word', '{}'))

    return tokenizer

# Load the tokenizer
with open('/Users/morgandixon/Desktop/1daystock_model_tokenizer.json') as f:
    json_data = f.read()
    tokenizer = tokenizer_from_json_string(json_data)
    return tokenizer

# Load the model
model = tf.keras.models.load_model('/Users/morgandixon/Desktop/1daystock_model', compile=False)

# Define the scaler object
scaler = MinMaxScaler()

# Data to string function
def data_to_string(data):
    data_str = ' '.join(map(str, data.flatten())).replace('\n', ' ').replace('  ', ' ').strip()
    return data_str

# Pre-process data function
def pre_process_data(data_list, labels, max_len, tokenizer):
    data_strings = [data_to_string(data) for data in data_list]
    sequences = tokenizer.texts_to_sequences(data_strings)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')
    labels = to_categorical(labels, num_classes=2)
    return sequences, labels

while True:
    # Get the stock symbol input from the user
    symbol = input("Enter the stock symbol (or 'exit' to quit): ")
    if symbol.lower() == 'exit':
        break
    try:
        stock_data = yf.download(symbol, period='1d', interval='5m')
    except Exception as e:
        print("Invalid stock symbol or an error occurred.")
        continue

    stock_data.dropna(inplace=True)
    if stock_data.empty:
        print("No data available for the given stock symbol.")
        continue

    # Select the features used during training
    selected_features = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
    selected_features = scaler.fit_transform(selected_features)
    max_len = 100
    sequences, _ = pre_process_data([selected_features], [0], max_len, tokenizer)
    predictions = model.predict(sequences)
    print(f"Raw predictions for {symbol}: {predictions}")

    # Convert predictions to labels
    labels = np.argmax(predictions, axis=1)
    if np.sum(labels) > 0.5 * len(labels):
        print(f"Predicted action for {symbol}: Buy")
    else:
        print(f"Predicted action for {symbol}: Sell")
