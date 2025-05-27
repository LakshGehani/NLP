import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from src.model import LSTMSentimentClassifier
from src.preprocess import preprocess_phrasebank
from src.glove_utils import load_glove_embeddings, build_embedding_matrix  # ✅ NEW

# Load data
X, y, vocab = preprocess_phrasebank("data/Sentences_75Agree.txt", max_length=30)
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42)

# Dataloaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GloVe and create embedding matrix
glove_path = "data/glove.6B.100d.txt"
glove_dict = load_glove_embeddings(glove_path)
embedding_matrix = build_embedding_matrix(vocab, glove_dict, embed_dim=100)

# Model with pretrained GloVe embeddings
model = LSTMSentimentClassifier(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    output_dim=3,
    padding_idx=0,
    pretrained_weights=embedding_matrix
).to(device)

# Weighted loss for class imbalance
weights = torch.tensor([0.3906, 0.1502, 0.4592]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=3, gamma=0.8)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    scheduler.step()

    print(f"Validation Accuracy: {correct/total:.2%}")

# Save model and vocab
torch.save(model.state_dict(), "models/lstm_sentiment_glove.pt")
with open("models/vocab_glove.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Model and vocab saved successfully.")
