import os
from typing import Tuple

import numpy as np
from grakel import Graph
from grakel.datasets.base import read_data
from grakel.kernels import EdgeHistogram, VertexHistogram
from scipy.sparse.csgraph import laplacian
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import (
    chi2_kernel,
    cosine_similarity,
    linear_kernel,
    rbf_kernel,
)
from sklearn.preprocessing import OneHotEncoder
from sympy import nextprime

from utils import get_prime_adjacency, get_prime_map_from_rel


def read_wrapper(
    path: str,
    name: str,
) -> Graph:
    """Helper function to load data in the tu-datasets form downloaded by hand.


    Args:
        path (str): Path to folder with data
        name (str): Name of the folder. Defaults to "MOLT-4".

    Returns:
        grakel.Graph: The dataset loaded.
    """
    cwd = os.getcwd()
    os.chdir(path)
    data = read_data(f"{name}")
    os.chdir(cwd)
    return data


### MODELS


class ProductPower(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        power: int = 1,
        aggr_str: str = "log",
        use_ohe: bool = False,
        use_laplace: bool = False,
        normalize: bool = False,
        grakel_compatible: bool = True,
        kernel_str: str = "rbf",
    ):
        """
        Graph Embedder working on Prime Adjacency Matrix (PAM).
        The outcome for each graph is a feature vector of shape (power,).
        Each cell k, 0<k<power contains the aggregated values of the PAM^k.
        The PAM^k values are aggregated according to the aggr_str. Currently:
        - prod: The product of all non-zero values
        - log: The sum of logs of all non-zero values

        Args:
            power (int, optional): The power of PAM up-to-which we calculate. Defaults to 1.
            aggr_str (str, optional): What type of aggregation to use. Currently supported ["prod", "log"].
                                      Defaults to "log" for overflowing.
            use_ohe (bool, optional): Whether to OHE the aggregated feature-values. Defaults to False.
            use_laplace (bool, optional): Whether to use the laplacian matrix of PAM. Defaults to False.
            normalize (bool, optional): Whether to normalize the result per aggregation. Defaults to False.
            grakel_compatible (bool, optional): Whether to create a kernel matrix (instead of a feature one). Defaults to True.
            kernel_str (str, optional): What type of kernel to use. Defaults to "rbf".

        Raises:
            AttributeError: _description_
        """
        self.power = power

        self.aggr_str = aggr_str
        if self.aggr_str == "log":
            self.aggr_fn = lambda x: np.sum(np.log(x))
        elif self.aggr_str == "prod":
            self.aggr_fn = np.prod
        self.use_ohe = use_ohe
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.use_laplace = use_laplace
        self.normalize = normalize
        self.grakel_compatible = grakel_compatible
        self.kernel_str = kernel_str
        if self.kernel_str == "rbf":
            self.kernel_fn = rbf_kernel
            self.gamma = 1
        elif self.kernel_str == "linear":
            self.kernel_fn = linear_kernel
        elif self.kernel_str == "cosine":
            self.kernel_fn = cosine_similarity
        elif self.kernel_str == "chi2":
            self.kernel_fn = chi2_kernel
        else:
            raise AttributeError(f"Kernel {self.kernel_str} is not understood!")
        self.x_train = None

    def fit(self, X, y=None):
        x_tr = []
        for x in X:
            if self.use_laplace:
                x = laplacian(x)
            try:
                _ = np.max(x)
            except ValueError:
                # this is for empty adjacency like TOX21_AR
                x_tr.append(np.array([0 for _ in range(self.power)]))
                continue
            cur_feat = []
            cur_x = x.copy()
            # Power = 0 (1-hop, original Adjacency)
            cur_prod = self.aggr_fn(cur_x[cur_x > 0])
            cur_feat.append(cur_prod)
            for _ in range(1, self.power):
                cur_x = np.matmul(cur_x, x)
                cur_prod = self.aggr_fn(cur_x[cur_x > 0])
                if self.normalize:
                    cur_prod -= cur_feat[-1]
                cur_feat.append(cur_prod)
            x_tr.append(np.array(cur_feat))
        x_tr = np.array(x_tr)
        if self.use_ohe:
            x_tr = self.ohe.fit_transform(x_tr).toarray()
        if self.grakel_compatible:
            self.x_train = x_tr
            self.gamma = 1 / (x_tr.shape[1] * self.x_train.var())
        return self

    def transform(self, X):
        x_tr = []
        for x in X:
            if self.use_laplace:
                x = laplacian(x)
            n = x.shape[0]
            if n == 0:
                x_tr.append(np.array([0 for _ in range(self.power)]))
                continue
            cur_feat = []
            cur_x = x.copy()
            # Power = 0 (1-hop, original Adjacency)
            cur_prod = self.aggr_fn(cur_x[cur_x > 0])
            cur_feat.append(cur_prod)
            for _ in range(1, self.power):
                cur_x = np.matmul(cur_x, x)
                cur_prod = self.aggr_fn(cur_x[cur_x > 0])
                if self.normalize:
                    cur_prod -= cur_feat[-1]
                cur_feat.append(cur_prod)
            x_tr.append(np.array(cur_feat))
        x_tr = np.array(x_tr)
        if self.use_ohe:
            x_tr = self.ohe.transform(x_tr)
        if self.grakel_compatible:
            if self.kernel_str == "rbf":
                x_tr = self.kernel_fn(x_tr, self.x_train, gamma=self.gamma)
            else:
                x_tr = self.kernel_fn(x_tr, self.x_train)
        return x_tr
