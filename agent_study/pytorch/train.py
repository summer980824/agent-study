
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


def main():
    from agent_study.pytorch.data import build_toy_dataset
    from agent_study.pytorch.model import TinyTextClassifier

    torch.manual_seed(42)
    vocab, X_train, y_train, X_test, y_test = build_toy_dataset()

    model = TinyTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=8,
        num_classes=2,
    )

    loss_history = train_model(model, X_train, y_train)

    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")

    model.eval()
    with torch.inference_mode():
        predictions = model(X_test).argmax(dim=1)
        accuracy = (predictions == y_test).float().mean()

    print(f"Test accuracy: {accuracy.item():.2%}")


if __name__ == "__main__":
    main()
