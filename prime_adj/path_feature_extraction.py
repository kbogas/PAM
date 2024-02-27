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
        pairs = [
            item for item in product(list(range(num_nodes)), list(range(num_nodes)))
        ]
    features = []
    for pair in pairs:
        cur_features = []
        for k_hop_pam in pam_powers:
            cur_features.append(k_hop_pam[pair[0], pair[1]])
        features.append(np.array(cur_features))
    feats = np.array(features)
    feats = pd.DataFrame(feats, columns=[f"val@{k+1}" for k in range(len(pam_powers))])
    feats["pairs"] = pairs
    return feats


if __name__ == "__main__":
    from data_loading import load_data
    from pam_creation import create_pam_matrices

    path = "./data/dummy_data"

    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path, project_name="test", add_inverse_edges="NO", sep=","
    )
    (
        pam_1hop_lossless,
        pam_powers,
        node2id,
        rel2id,
        broke_cause_of_sparsity,
    ) = create_pam_matrices(
        df_train,
        max_order=5,
        method="plus_times",
        use_log=False,
        eliminate_diagonal=False,
        spacing_strategy="step_1",
    )
    path_feats = generate_path_features(pam_powers, pairs=[])
    print(path_feats)
