import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model
model_dir = "models/finbert-finetuned"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Sentences to test
sentences = [
    "Apple stock plunges after missing earnings expectations.",
    "Apple shares surge after record-breaking iPhone sales.",
    "Apple reports Q2 earnings in line with analyst forecasts.",
    "Analysts remain uncertain about Apple's growth outlook.",
    "Strong demand for MacBooks lifts Apple's profit margins."
]

label_map = {0: "positive", 1: "neutral", 2: "negative"}

# Run inference
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
        probs = F.softmax(output.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    print(f"\nSentence: {sentence}")
    print(
        f"Predicted Sentiment: {label_map[pred]}")
