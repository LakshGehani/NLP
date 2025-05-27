import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_phrasebank
import torch

X, y, vocab = preprocess_phrasebank("data/Sentences_AllAgree.txt", max_length=30)

# Convert to torch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

print("Shape:", X_tensor.shape)  # [N, max_len]
print("Sample:", X_tensor[0])
print("Label:", y_tensor[0])
print("Vocab size:", len(vocab))
