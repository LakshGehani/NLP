import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from src.model import LSTMSentimentClassifier
from src.preprocess import preprocess_phrasebank


# Load data
X, y, vocab = preprocess_phrasebank("data/Sentences_AllAgree.txt", max_length=30)
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Split train/test
X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42)

# Dataloaders
batch_size = 32
train_loader = DataLoader(TensorDataset(
    X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentimentClassifier(vocab_size=len(vocab)).to(device)
loss_fn = nn.CrossEntropyLoss()
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


# Save model weights
torch.save(model.state_dict(), "models/lstm_sentiment.pt")

# Save vocab as pickle
with open("models/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Model and vocab saved successfully.")
