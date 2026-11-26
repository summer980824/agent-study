import torch


def simple_attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights
