"""Load the trained model and score any review you type. No retraining needed."""
import os
import sys

from tensorflow.keras.models import load_model

from utils import MODEL_PATH, make_predictor


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"'{MODEL_PATH}' not found. Run `python train.py` first.")

    model = load_model(MODEL_PATH)
    predict = make_predictor(model)

    print("Sentiment analyzer ready. Type a review (or 'quit' to exit).")
    while True:
        text = input("\nReview: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        score, label, confidence = predict(text)
        print(f"Sentiment : {label}")
        print(f"Score     : {score:.3f}  (0 = very negative, 1 = very positive)")
        print(f"Rating    : {score * 10:.1f} / 10")
        print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
