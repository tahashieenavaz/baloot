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

    largest_values, largest_indices = baloot.k_largest(
        universe,
        2,
        with_index=True,
    )

    assert torch.allclose(largest_values, torch.tensor([77, 9]))
    assert torch.allclose(largest_indices, torch.tensor([1, 6]))


def test_k_largest_returns_sorted_values():
    universe = torch.tensor([5, 100, 20, 3, 80])

    largest = baloot.k_largest(universe, 3)

    assert torch.allclose(largest, torch.tensor([100, 80, 20]))


def test_k_largest_with_k_equal_to_one():
    universe = torch.tensor([10, 50, 20])

    largest = baloot.k_largest(universe, 1)

    assert torch.allclose(largest, torch.tensor([50]))


def test_k_largest_with_negative_values():
    universe = torch.tensor([-10, -2, -7, -1])

    largest = baloot.k_largest(universe, 2)

    assert torch.allclose(largest, torch.tensor([-1, -2]))


def test_k_largest_handles_duplicates():
    universe = torch.tensor([5, 5, 3, 1])

    largest = baloot.k_largest(universe, 2)

    assert torch.allclose(largest, torch.tensor([5, 5]))


def test_k_largest_does_not_modify_input():
    universe = torch.tensor([4, 10, 2])
    original = universe.clone()

    baloot.k_largest(universe, 2)

    assert torch.equal(universe, original)


def test_k_largest_does_not_accept_index_and_with_index_together():
    universe = torch.tensor([-4, 77, 2, 3, 4, 5, 9, 0])

    with pytest.raises(ValueError):
        baloot.k_largest(
            universe,
            2,
            with_index=True,
            index=True,
        )


def test_k_largest_rejects_k_zero():
    universe = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError):
        baloot.k_largest(universe, 0)


def test_k_largest_rejects_negative_k():
    universe = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError):
        baloot.k_largest(universe, -1)


def test_k_largest_rejects_k_larger_than_tensor():
    universe = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError):
        baloot.k_largest(universe, 10)


def test_k_largest_rejects_empty_tensor():
    universe = torch.tensor([])

    with pytest.raises(ValueError):
        baloot.k_largest(universe, 1)
