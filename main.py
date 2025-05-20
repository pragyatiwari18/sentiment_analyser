import re
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ensure NLTK data
nltk.download("stopwords")
nltk.download("vader_lexicon")

app = Flask(__name__)

model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# initialize VADER
vader = SentimentIntensityAnalyzer()

def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    # 1) model prediction
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=100)
    prob = float(model.predict(pad)[0][0])
    model_sent = "Positive" if prob > 0.5 else "Negative"

    # 2) VADER fallback
    vader_scores = vader.polarity_scores(text)
    # if VADER is strongly negative, override
    if vader_scores["compound"] < -0.2:
        return "Negative", vader_scores["compound"]
    # if VADER is strongly positive, override
    if vader_scores["compound"] > 0.2:
        return "Positive", vader_scores["compound"]

    # otherwise trust LSTM
    return model_sent, prob

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    if request.method == "POST":
        t = request.form.get("text")
        if t:
            result, confidence = predict_sentiment(t)
            confidence = round(confidence, 2)
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
