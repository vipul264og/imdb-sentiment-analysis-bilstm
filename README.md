# IMDB Sentiment Analysis with a Bidirectional LSTM

Binary sentiment classifier (positive / negative) for movie reviews, built with TensorFlow/Keras on the IMDB dataset (50,000 reviews).

## Model

| Layer | Details |
|-------|---------|
| Embedding | 10,000-word vocab, 64 dims, padding masked |
| Bidirectional LSTM | 64 units |
| Dropout | 0.5 |
| Dense | 32 units, ReLU |
| Dropout | 0.3 |
| Dense | 1 unit, sigmoid |

Reviews are truncated/padded to 200 tokens. Training uses Adam, binary cross-entropy, and early stopping on validation loss.

## Results

Fill these in after running:

- Test accuracy: `84.5%`
- Training curves: see `training_curves.png`

## Run it

```bash
pip install -r requirements.txt

python train.py     # train once; saves sentiment_lstm.keras
python predict.py   # type any review, get sentiment instantly
```

Example session:

```
Review: The plot was dull but the acting was brilliant
Sentiment : Positive 😊
Score     : 0.712  (0 = very negative, 1 = very positive)
Rating    : 7.1 / 10
Confidence: 71.2%
```

`train.py` also prints a classification report and saves `training_curves.png`.
`predict.py` only loads the saved model, so no retraining is needed.

## Ideas for future work

- Compare with GRU and a 1D-CNN
- Use pretrained embeddings (GloVe)
- Fine-tune a transformer (e.g. DistilBERT) and compare accuracy
- Wrap the predictor in a small Streamlit or Gradio app

## License

MIT, see [LICENSE](LICENSE).
