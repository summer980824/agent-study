import torch
from torch import nn

def build_toy_dataset():
    torch.manual_seed(42)
    vocab = {
        'good': 0,
        'bad': 1,
        'great': 2,
        'awful': 3,
        'delicious': 4,
        'terrible': 5,
        'excellent': 6,
        'poor': 7,
        'movie': 8,
        'food': 9,
        'service': 10,
    }

    X_train = torch.tensor([
        [vocab['good'], vocab['movie']],
        [vocab['bad'], vocab['movie']],
        [vocab['great'], vocab['food']],
        [vocab['awful'], vocab['food']],
        [vocab['delicious'], vocab['service']],
        [vocab['terrible'], vocab['service']],
        [vocab['excellent'], vocab['movie']],
        [vocab['poor'], vocab['food']],
    ], dtype=torch.long)

    y_train = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])

    X_test = torch.tensor([
        [vocab['good'], vocab['food']],
        [vocab['bad'], vocab['service']],
    ], dtype=torch.long)

    y_test = torch.tensor([1, 0])
    return vocab, X_train, y_train, X_test, y_test

