import torch
import numpy as np


def load_glove_embeddings(filepath="data/glove.6B.100d.txt"):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def build_embedding_matrix(vocab, glove_dict, embed_dim=100):
    matrix = np.zeros((len(vocab), embed_dim))
    for word, idx in vocab.items():
        if word in glove_dict:
            matrix[idx] = glove_dict[word]
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
    return torch.tensor(matrix, dtype=torch.float32)
