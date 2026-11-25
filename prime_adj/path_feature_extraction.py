from itertools import product

import numpy as np
import pandas as pd
from scipy.sparse import csr_array


def generate_path_features(
    pam_powers: list[csr_array],
    pairs: list[tuple[int, int]] = [],
) -> pd.DataFrame:
    """ "
    Function to generate path features for the pairs given.
    (Or all pairs if no list of pairs is given).

    Args:
        pam_powers (list[csr_array]): List of sparse arrays.
        pairs (list[tuple[int, int]], optional): Pair-ids for which to generate the features. If none given
        all are generated. Defaults to [].

    Returns:
        pd.DataFrame:
    """

    num_nodes = pam_powers[0].shape[0]
    if len(pairs) == 0:
        pairs = product(list(range(num_nodes)), list(range(num_nodes)))
    features = []
    pair_names = []
    for pair in pairs:
        cur_features = []
        for k_hop_pam in pam_powers:
            cur_features.append(k_hop_pam[pair[0], pair[1]])
        features.append(np.array(cur_features))
        pair_names.append(pair)
    feats = np.array(features)
    feats = pd.DataFrame(feats, columns=[f"val@{k+1}" for k in range(len(pam_powers))])
    feats["pairs"] = pair_names
    return feats
