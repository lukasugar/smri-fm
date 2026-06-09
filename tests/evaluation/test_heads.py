import pytest
import torch

from evaluation.heads import LinearHead


def test_linear_head_first_pooling():
    head = LinearHead(input_dim=3, output_dim=1, pooling="first")
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    tokens = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])

    assert torch.equal(head(tokens), torch.tensor([[6.0]]))


def test_linear_head_mean_pooling():
    head = LinearHead(input_dim=2, output_dim=1, pooling="mean")
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    tokens = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    assert torch.equal(head(tokens), torch.tensor([[8.0]]))


def test_linear_head_requires_token_sequence():
    head = LinearHead(input_dim=2, output_dim=1, pooling="mean")

    with pytest.raises(ValueError, match="\\[B, T, D\\]"):
        head(torch.zeros(2, 2))


def test_linear_head_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="unknown pooling"):
        LinearHead(input_dim=2, output_dim=1, pooling="max")
