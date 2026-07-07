import baloot
import torch


def test_arccot_float_and_int_and_tensor_match():
    with_integer = baloot.arccot(4)
    with_float = baloot.arccot(4.0)
    with_tensor = baloot.arccot(torch.tensor(4.0))

    assert torch.allclose(with_integer, with_tensor)
    assert torch.allclose(with_float, with_tensor)


def test_calculates_arccot_correctly():
    difference = abs(baloot.arccot(2).item() - 0.46365)
    assert difference < pow(10, -4)
