import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

base_path = "/Users/morgandixon/Desktop/stockinfo"
num_classes = 2
batch_size = 64
epochs = 30
model_file_name = "stock_classifier.pb"
labels = ["Buy", "Sell"]
train_data = []
train_labels = []
test_data = []
test_labels = []

train_data_count = 0
test_data_count = 0

for label in labels:
    train_data_path = os.path.join(base_path, label, "train")
    test_data_path = os.path.join(base_path, label, "test")

    train_data_files = os.listdir(train_data_path)
    for file_name in train_data_files:
        file_path = os.path.join(train_data_path, file_name)
        print(file_path)
        data = pd.read_csv(file_path, header=None, encoding='ISO-8859-1').iloc[:, 1:].values
        train_data.append(data)
        train_labels.append(labels.index(label))
        train_data_count += 1  # Increment train data count

    test_data_files = os.listdir(test_data_path)
    for file_name in test_data_files:
        file_path = os.path.join(test_data_path, file_name)
        print(file_path)
        data = pd.read_csv(file_path, header=None, encoding='ISO-8859-1').iloc[:, 1:].values
        test_data.append(data)
        test_labels.append(labels.index(label))
        test_data_count += 1  # Increment test data count

print(f"Total train data files: {train_data_count}")
print(f"Total test data files: {test_data_count}")

def data_to_string(data):
    data_str = ' '.join(map(str, data.flatten())).replace('\n', ' ').replace('  ', ' ').strip()
    return data_str

# Tokenize the strings into characters
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

# Define the data_to_string function
def data_to_string(data):
    data_str = ' '.join(map(str, data.flatten())).replace('\n', ' ').replace('  ', ' ').strip()
    return data_str

# Create a list of texts from the train_data
texts = [data_to_string(data) for data in train_data]

# Initialize the tokenizer
tokenizer = Tokenizer(filters='', lower=False, split='\n')
tokenizer.fit_on_texts(texts)

# Save the tokenizer as a JSON file
tokenizer_json = tokenizer.to_json()
with open('/Users/morgandixon/Desktop/1daystock_model_tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Truncate the sequences to a fixed length
max_sequence_length = 100

# Preprocess data by converting CSV data to strings and tokenize them
def pre_process_data(data_list, labels, tokenizer):
    data_strings = [data_to_string(data) for data in data_list]
    tokenizer.fit_on_texts(data_strings)
    sequences = tokenizer.texts_to_sequences(data_strings)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post', dtype='int32')
    labels = to_categorical(labels, num_classes=num_classes)
    return sequences, labels

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Preprocess the data
x_train_processed, y_train_processed = pre_process_data(x_train, y_train, tokenizer)
x_val_processed, y_val_processed = pre_process_data(x_val, y_val, tokenizer)

# Define the model
def create_model():
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 32

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_model()
# Compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_processed, y_train_processed, epochs=epochs, batch_size=batch_size, validation_data=(x_val_processed, y_val_processed))

# Save model
model_file_name = "/Users/morgandixon/Desktop/1daystock_model"
model.save(model_file_name, save_format="tf")
