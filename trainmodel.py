import numpy as np
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure NLTK stopwords are available
nltk.download("stopwords")

# === Step 1: Load and label data from input.txt ===
def load_data():
    texts, labels = [], []
    current_label = None

    with open("input.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Positive" in line:
                current_label = 1
            elif "Negative" in line:
                current_label = 0
            elif line:
                texts.append(line)
                labels.append(current_label)
    return texts, labels

# === Step 2: Clean the text ===
def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# === Step 3: Load tokenizer ===
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

texts, labels = load_data()
cleaned_texts = [clean_text(text) for text in texts]

# === Step 4: Convert texts to sequences ===
X = tokenizer.texts_to_sequences(cleaned_texts)
X = pad_sequences(X, maxlen=100)
y = np.array(labels)

# === Step 5: Build and train the LSTM model ===
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

print("Training model...")
model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2)

# === Step 6: Save the model ===
model.save("lstm_model.h5")
print("✅ Model saved as lstm_model.h5")
