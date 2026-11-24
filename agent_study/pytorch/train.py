
import torch
from torch import nn

def train_model(
    model,
    X_train,
    y_train,
    epochs=101,
    learning_rate=0.1,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = [0.0] * epochs

    for epoch in range(epochs):
        model.train()

        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[epoch] = loss.item()

    return loss_history
