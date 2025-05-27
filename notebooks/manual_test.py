import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import pickle
from src.model import LSTMSentimentClassifier
from src.preprocess import clean_text, encode_tokens, pad_sequences

# --- Load vocab and model ---
with open("models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = LSTMSentimentClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load("models/lstm_sentiment.pt"))
model.eval()

# --- Label map ---
label_map = {0: "positive", 1: "neutral", 2: "negative"}

# --- Input your test sentences ---
sentences = [
    "Apple stock plunges after missing earnings expectations.",
    "Apple shares surge after record-breaking iPhone sales.",
    "Apple reports Q2 earnings in line with analyst forecasts.",
    "Analysts remain uncertain about Apple's growth outlook.",
    "Strong demand for MacBooks lifts Apple's profit margins."
]

# --- Predict sentiment ---
for sentence in sentences:
    tokens = clean_text(sentence)
    encoded = encode_tokens(tokens, vocab)
    padded = pad_sequences([encoded], max_length=30)
    x_tensor = torch.tensor(padded)

    with torch.no_grad():
        output = model(x_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    print(f"\nSentence: {sentence}")
    print(f"Predicted Sentiment: {label_map[pred]} ({confidence:.2f} confidence)")
