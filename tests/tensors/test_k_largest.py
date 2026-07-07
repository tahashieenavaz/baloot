import baloot
import torch
import pytest


def test_finds_2_largest_values():
    universe = torch.tensor([-4, 77, 2, 3, 4, 5, 9, 0])
    largest = baloot.k_largest(universe, 2)
    assert torch.allclose(largest, torch.tensor([77, 9]))


def test_finds_2_largest_indices():
    universe = torch.tensor([-4, 77, 2, 3, 4, 5, 9, 0])
    largest_indices = baloot.k_largest(universe, 2, index=True)
    assert torch.allclose(largest_indices, torch.tensor([1, 6]))


def test_finds_2_largest_indices_and_values():
    universe = torch.tensor([-4, 77, 2, 3, 4, 5, 9, 0])
    largest_values, largest_indices = baloot.k_largest(universe, 2, with_index=True)
    assert torch.allclose(largest_values, torch.tensor([77, 9]))
    assert torch.allclose(largest_indices, torch.tensor([1, 6]))


def test_k_largest_does_not_accept_index_and_with_index_together():
    universe = torch.tensor([-4, 77, 2, 3, 4, 5, 9, 0])

    with pytest.raises(ValueError):
        baloot.k_largest(universe, 2, with_index=True, index=True)
