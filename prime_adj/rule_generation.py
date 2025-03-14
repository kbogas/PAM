import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import csr_array

from prime_adj.utils import get_primefactors


def create_ruleset(
    pam_1hop_lossless: csr_array,
    pam_powers: list[csr_array],
    use_log: bool,
    max_num_hops: int = -1,
    laplace_smoothing_alpha: float = 1,
) -> pd.DataFrame:
    """
    Simple function to generate rules/metapaths based on look-up procedure
    of non-zero overlapping elements in the k-hop matrices and the original
    graph. Currently only "same-cell" rules are generated. These means
    that only cyclic path rules of the form
    r1(A,B) ^ r2(B,C) ^ r3(C,D) => r_head(A,D) can be created.
    For more details see:
    https://arxiv.org/abs/2305.10467

    Args:
        pam_powers (csr_array): Lossless 1-hop
        pam_powers (list[csr_array]): List of the k-hop pams.
        use_log (bool): Whether the primes are logarithmed first.
        max_num_hops (int): Maximun hops to take into account when creating rules.
        Range in [1, len(pam_powers) + 1]
        laplace_smoothing_alpha (int): Laplace smoothing for confidence of rules

    Returns:
        pd.DataFrame: DataFrame with rules and details
    """

    # This will hold the whole rule-set
    all_rules = []
    # Iterate over all k-hops (including the 1-hop power)
    if max_num_hops == -1:
        max_num_hops = len(pam_powers) + 1
    k_hop_pams_to_mine = [pam_1hop_lossless] + pam_powers[1:]
    k_hop_pams_to_mine = k_hop_pams_to_mine[:max_num_hops]
    for k_hop_index, k_hop_pam in tqdm.tqdm(
        enumerate(k_hop_pams_to_mine), total=len(k_hop_pams_to_mine)
    ):
        # The rules that are extracted at this hop level
        cur_rules = []
        # These are the aligned non-zero elements between the 1-hop and the k-hop matrix
        # These indicate a probable rule k-hop chain -> head rel
        overlapping_nnz_rows, overlapping_nnz_cols = (
            k_hop_pam.multiply(pam_1hop_lossless)
        ).nonzero()
        # If there is actually any overlapping element
        if overlapping_nnz_rows.shape[0] > 0:
            # Keep the direct (1-hop) relations in these overlapping data
            overlapping_1hop_vals = pam_1hop_lossless[
                overlapping_nnz_rows, overlapping_nnz_cols
            ]
            # Iterarate over these direct relations
            for overlapping_element_index, overlapping_1hop_val in enumerate(
                overlapping_1hop_vals
            ):

                # Keep track of the corresponding k-hop semantic chain value
                k_hop_cell_value = k_hop_pam[
                    overlapping_nnz_rows[overlapping_element_index],
                    overlapping_nnz_cols[overlapping_element_index],
                ]
                # Factorize them in case there are more than one direct relations
                head_primes = get_primefactors(overlapping_1hop_val)
                # Iterate over these head relations that are implied by the k-hop chain
                # And create a rule for each one of them
                for head_prime in head_primes:
                    # If we are at hop 1, decompose the product of relations if available
                    if k_hop_index == 0:
                        k_hop_values = get_primefactors(k_hop_cell_value)
                        if use_log:
                            k_hop_values = [np.log(khv) for khv in k_hop_values]
                    # else use the whole k-hop chain as is
                    else:
                        k_hop_values = [k_hop_cell_value]
                    for k_hop_value in k_hop_values:
                        # This is so we do not have 7->7 type of rules from the 1-hop case
                        if head_prime != k_hop_value:

                            cur_rules.append(
                                {
                                    "head_rel": head_prime,
                                    "body_chain": k_hop_value,
                                    "num_hops": k_hop_index,
                                    "type": "same_cell",
                                }
                            )
            # If any rules were mined at this level
            if cur_rules:
                # The following process is to capture some support and coverage statistics
                # Because we may have many occurrences of the the rule
                df_rules = pd.DataFrame(cur_rules)

                grouped_by_head_and_body = (
                    df_rules.groupby(["head_rel", "type"])["body_chain"]
                    .value_counts()
                    .to_dict()
                )
                # This expresses the times each head relation is implied (at this k-hop)
                # For the rule 15=>5 at 2-hop, this counts the times that 5 is implied
                head_support = df_rules["head_rel"].value_counts().to_dict()
                # This expresses the times each body value is implied (e.g. 15)
                # For the rule 15=>5 at 2-hop, this counts the times that 15 is found in the 2-hop matrix
                if k_hop_index == 0:
                    bodies_to_use = [
                        item
                        for product in k_hop_pam.data
                        for item in get_primefactors(product)
                    ]
                    if use_log:
                        bodies_to_use = np.log(bodies_to_use)
                else:
                    bodies_to_use = k_hop_pam.data
                body_support = pd.Series(bodies_to_use).value_counts().to_dict()
                for (head_value, type_, body_value) in grouped_by_head_and_body:  # type: ignore
                    all_rules.append(
                        {
                            "head_rel": head_value,
                            "body": body_value,
                            "head_body_count": grouped_by_head_and_body[
                                (head_value, type_, body_value)
                            ],
                            "body_count": body_support[body_value],
                            "head_count": head_support[head_value],
                            "hop": k_hop_index + 1,
                            "type": type_,
                        }
                    )
    # Finally calculate some statistics, namely confidence, head_coverage
    # and the F1 score of these two
    all_rules_df = pd.DataFrame(all_rules)
    all_rules_df["conf"] = all_rules_df["head_body_count"] / (
        all_rules_df["body_count"] + laplace_smoothing_alpha
    )
    all_rules_df["head_coverage"] = all_rules_df["head_body_count"] / (
        all_rules_df["head_count"]
    )
    all_rules_df["score"] = 2 * (
        all_rules_df["conf"]
        * all_rules_df["head_coverage"]
        / (all_rules_df["head_coverage"] + all_rules_df["conf"])
    )
    all_rules_df = all_rules_df.sort_values(["score"], ascending=False)

    if any(all_rules_df["body"] < 0):
        print(
            f"Negative body values as rule patterns occured. This is usually due to numeric overflow. \
            Please consider using 'use_log=True' for this project..."
        )

    if use_log:
        all_rules_df["head_rel"] = all_rules_df["head_rel"].apply(np.log)

    return all_rules_df


if __name__ == "__main__":
    from data_loading import load_data
    from pam_creation import create_pam_matrices

    path = "./data/dummy_data"

    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path, project_name="test", add_inverse_edges="NO", sep=",", skiprows=0
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
        spacing_strategy="step_1",
    )
    df_rules = create_ruleset(
        pam_1hop_lossless, pam_powers, use_log=False, max_num_hops=-1
    )
    print(df_rules)
