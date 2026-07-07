import baloot
import torch


def test_calculates_trace_of_a_simple_matrix():
    matrix = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )

    assert baloot.trace(matrix) == 10


def test_calculates_the_trace_of_a_batch_of_same_size_matrices():
    matrices = torch.tensor(
        [
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            [
                [767, 410, 972, 877],
                [767, 410, 972, 877],
                [767, 410, 972, 877],
                [767, 410, 972, 877],
            ],
        ]
    )

    traces = baloot.trace(matrices)

    assert traces[0] == 10
    assert traces[1] == 3026
