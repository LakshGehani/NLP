import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return tokens


def build_vocab(token_lists, min_freq=2):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_tokens(token_list, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in token_list]


def pad_sequences(sequences, pad_value=0, max_length=None):
    if not max_length:
        max_length = max(len(seq) for seq in sequences)
    padded = [seq[:max_length] + [pad_value] *
              (max_length - len(seq)) for seq in sequences]
    return padded


def preprocess_phrasebank(file_path, min_freq=2, max_length=50):
    texts, labels = [], []
    label_map = {"positive": 0, "neutral": 1, "negative": 2}

    with open(file_path, encoding="latin-1") as f:
        for line in f:
            if "@" not in line:
                continue
            try:
                text, label = line.strip().rsplit("@", 1)
                tokens = clean_text(text.strip())
                texts.append(tokens)
                labels.append(label_map.get(label.lower(), -1))
            except ValueError:
                continue

    # Remove invalid labels
    df = pd.DataFrame({"tokens": texts, "label": labels})
    df = df[df["label"] != -1]

    vocab = build_vocab(df["tokens"].tolist(), min_freq=min_freq)
    encoded = [encode_tokens(tokens, vocab) for tokens in df["tokens"]]
    padded = pad_sequences(encoded, pad_value=0, max_length=max_length)

    return padded, df["label"].tolist(), vocab
