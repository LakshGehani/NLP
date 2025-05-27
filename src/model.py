import torch
import torch.nn as nn


class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # x2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)                              # [B, T, E]
        # hidden: [2, B, H]
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate final forward & backward hidden
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        # [B, output_dim]
        return self.fc(self.dropout(hidden))
