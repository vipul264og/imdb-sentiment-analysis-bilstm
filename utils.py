"""Shared settings and text-encoding helpers."""
import re

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LENGTH = 200
INDEX_FROM = 3  # Keras reserves 0=pad, 1=start, 2=unknown
MODEL_PATH = "sentiment_lstm.keras"


def make_predictor(model):
    """Return a function that scores any text with the given model."""
    word_index = imdb.get_word_index()

    def encode_review(text):
        text = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
        encoded = [1]  # start token, as in the training data
        for word in text.split():
            idx = word_index.get(word)
            idx = idx + INDEX_FROM if idx is not None else 2
            encoded.append(idx if idx < VOCAB_SIZE else 2)
        return encoded

    def predict_sentiment(text):
        padded = pad_sequences(
            [encode_review(text)], maxlen=MAX_LENGTH, padding="post", truncating="post"
        )
        score = float(model.predict(padded, verbose=0)[0][0])
        label = "Positive 😊" if score >= 0.5 else "Negative 😞"
        confidence = score if score >= 0.5 else 1 - score
        return score, label, confidence

    return predict_sentiment
