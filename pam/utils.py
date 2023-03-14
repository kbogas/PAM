import random

import numpy as np
from scipy.sparse import csr_matrix


def set_all_seeds(seed: int = 0):
    """Fix random seeds

    Args:
        seed (int): Random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    return 1


def get_sparsity(A: csr_matrix) -> float:
    """Calculate sparsity % of scipy sparse matrix.

    Args:
        A (scipy.sparse): Scipy sparse matrix

    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] ** 2))
