"""Train the Bidirectional LSTM on IMDB and save it. Run once."""
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import MAX_LENGTH, MODEL_PATH, VOCAB_SIZE

EMBED_DIM = 64

tf.keras.utils.set_random_seed(42)


def load_data():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding="post", truncating="post")
    return X_train, y_train, X_test, y_test


def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history, path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
    )
    plot_history(history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f} | Test loss: {loss:.4f}")

    y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).ravel()
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    model.save(MODEL_PATH)

    print(f"\nModel saved to {MODEL_PATH}. Now run: python predict.py")


if __name__ == "__main__":
    main()
