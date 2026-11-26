import torch
from torch import nn

class TinyTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        pooled = embedded.mean(dim=1)
        return self.classifier(pooled)
