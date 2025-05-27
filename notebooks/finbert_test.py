import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# --- Load fine-tuned model and tokenizer ---
model_dir = "models/finbert-finetuned"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# --- Load your raw news headlines ---
# Adjust if filename changed
df = pd.read_csv("data/raw_news_PLTR_2025-05-27.csv")
texts = df['title'].astype(str).tolist()

# --- Predict sentiment ---
preds = []
confidences = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        preds.append(pred)
        confidences.append(round(confidence, 4))

# --- Map labels ---
label_map = {0: "positive", 1: "neutral", 2: "negative"}
df['finbert_pred'] = preds
df['finbert_label'] = df['finbert_pred'].map(label_map)
df['finbert_confidence'] = confidences

# --- Save the result ---
df.to_csv("data/sentiment_news_PLTR_finetuned_finbert.csv", index=False)
print("[✔] Predictions saved to: data/sentiment_news_PLTR_finetuned_finbert.csv")
