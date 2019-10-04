import pytest
import numpy as np

from wormbrain.match import pairwise_distance


def test_pairwise_distance():
    """
    A test for computing pairwise distances between two sets of points.
    """
    A = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    B = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    D = pairwise_distance(A, B)

    assert D.ndim == 2
    assert D.shape == (2, 2)
    assert np.allclose(D,  [[4., 0.], [5.65685425, 4.]])

