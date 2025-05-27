import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text
import pandas as pd

def prepare_finbert_csv(input_path, output_path):
    texts = []
    labels = []
    label_map = {"positive": 0, "neutral": 1, "negative": 2}

    with open(input_path, encoding="latin-1") as f:
        for line in f:
            if "@" not in line:
                continue
            try:
                text, label = line.strip().rsplit("@", 1)
                text = text.strip()
                label = label_map.get(label.strip().lower(), -1)
                if label != -1:
                    texts.append(text)
                    labels.append(label)
            except ValueError:
                continue

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(output_path, index=False)
    print(f"FinBERT training CSV saved to: {output_path}")


# Example usage
prepare_finbert_csv("data/Sentences_75Agree.txt", "data/finbert_train.csv")
