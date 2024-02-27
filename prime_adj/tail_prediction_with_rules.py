import math
import string

import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import csr_array

from prime_adj.utils import get_primefactors_multiplicity


def noisy_or_aggregator(confidences: list):
    """Noisy or aggregator according to AnyBURL
    for aggregating multiple rule confidences in one score.
    This is a strategy to merge the confidence of multiple
    rules that were triggered for the same prediction into
    one score.

    Args:
        confidences (list): List of scores.

    Returns:
        float: The aggregated score.
    """
    return 1 - np.prod([1 - confindence for confindence in confidences])


def generate_rule_string(
    chain_of_relations: list[str], head_rel: str
) -> tuple[str, str, str]:
    """Helper function to create the rule

    Args:
        chain_of_relations (list[str]): The chain of relations followed.
        head_rel (str): The relation implied in the head.

    Returns:
        tuple[str, str, str]: The string of the rule, the first letter (A) and the last
        letter (B). The last two values are there to help change the final result.
    """

    possible_letters = string.ascii_uppercase
    first_letter = possible_letters[0]
    rule_str = ""
    for i_rel, relation in enumerate(chain_of_relations):
        next_letter = possible_letters[possible_letters.index(first_letter) + 1]
        rule_str += f"{relation}( {first_letter} , {next_letter} ) ^ "
        first_letter = next_letter

    rule_str = rule_str[:-2]
    head = f"-> {head_rel}( {possible_letters[0]} , {next_letter} )"

    rule_str += head
    return rule_str, possible_letters[0], next_letter


def map_dataframe(
    df: pd.DataFrame,
    node2id: dict[str, int],
    rel2id: dict[str, int],
    dropna: bool = True,
) -> pd.DataFrame:
    """Helper function that maps node and relation names to integers.

    Args:
        df (pd.DataFrame): The original dataframe
        node2id (dict[str, int]): The dictionary mapping nodes to their ids e.g. "Ent1"->1.
        rel2id (dict[str, int]): The dictionary mapping relations to their ids e.g. "rel1"->3.
        dropna (bool): If True, drops the unmapped rows.

    Returns:
        pd.DataFrame: The mapped dataframe. If any could not be mapped, are simply dropped.
    """
    df_mapped = df.copy()
    df_mapped["rel"] = df["rel"].map(rel2id)
    df_mapped["head"] = df["head"].map(node2id)
    df_mapped["tail"] = df["tail"].map(node2id)

    if dropna:
        # print(f"Length before dropping nan", len(df_mapped))
        df_mapped.dropna(inplace=True)
        # print(f"Length after dropping nan", len(df_mapped))
    df_mapped["head"] = df_mapped["head"].astype(int)
    df_mapped["tail"] = df_mapped["tail"].astype(int)
    return df_mapped


def predict_tail_with_explanations(
    df_test: pd.DataFrame,
    all_rules_df: pd.DataFrame,
    pam_powers: list[csr_array],
    node2id: dict[str, int],
    rel2id: dict[str, int],
    rank_rules_by_: str = "score",
) -> None:
    """_summary_

    Args:
        df_test (pd.DataFrame): The dataframe with columns ['head', 'rel', 'tail'] containing the test triples.
        all_rules_df (pd.DataFrame): The dataframe with the rules as generated by rule_generation.py.
        pam_powers (list[csr_array]): The PAM powers to check for triggered rules.
        node2id (dict[str, int]): The dictionary mapping nodes to their ids e.g. "Ent1"->1.
        rel2id (dict[str, int]): The dictionary mapping relations to their ids e.g. "rel1"->3
        rank_rules_by_ (str): The attribute over which we decide the importance of the rule
    Returns:
        Nothing. Simply prints each prediction and exits
    """
    # Map each node to the corresponding PAM index ID
    # And each relationship to its prime number
    df_test_mapped = map_dataframe(df_test, node2id, rel2id)

    # Create the inverse dicts as well for printing purposes
    id2rel = dict(zip(rel2id.values(), rel2id.keys()))
    id2node = dict(zip(node2id.values(), node2id.keys()))

    # Iterate over each query
    for test_index, test_row in df_test_mapped.iterrows():

        probable_predictions = 0
        query_head = test_row["head"]
        query_rel = test_row["rel"]
        query_tail = test_row["tail"]
        print(
            f"Query ({test_index}): {id2node[query_head]} - {id2rel[query_rel]} - {id2node[query_tail]}"
        )

        # Keep only the rules that imply this relationship
        query_rules_all_hops = all_rules_df[all_rules_df["head_rel"] == query_rel]
        # Group them by hop-level (i.e. all rules of length 2, all rules of length 3 etc.)
        hop_level_to_rule_body_values = (
            query_rules_all_hops.groupby("hop")["body"].aggregate(list).to_dict()
        )
        # Iterate at each hop-level to find possible matches
        for k_hop_level, body_values in hop_level_to_rule_body_values.items():

            # Keep track of the confidence that each rule has
            k_hop_rules_conf = query_rules_all_hops[
                query_rules_all_hops["hop"] == k_hop_level
            ][["body", rank_rules_by_]]
            k_hop_rules_conf_dict = dict(
                zip(k_hop_rules_conf["body"], k_hop_rules_conf[rank_rules_by_])
            )
            # Find the corresponding row for this query head from the appropriate
            # k-hop matrix
            k_hop_pam_head_chains = pam_powers[k_hop_level - 1][[query_head]]
            # Find where the nnz entries of this row, match any of the rules that
            # we have recongized
            probable_tail_entities_indices = np.where(
                np.in1d(k_hop_pam_head_chains.data, body_values)
            )[0]
            # If there exist such cases, then some rules are triggered and these
            # lead us to predictions
            if probable_tail_entities_indices.shape[0] > 0:
                # Keep note of the possible tails
                probable_tail_entities_nodes = k_hop_pam_head_chains.indices[
                    probable_tail_entities_indices
                ]
                # Keep note of the rules that generated each tail
                probable_tail_triggered_rules = k_hop_pam_head_chains.data[
                    probable_tail_entities_indices
                ]
                # Keep note of the confidence of each rule
                probable_tail_triggered_rules_confidences = np.array(
                    [
                        k_hop_rules_conf_dict[rule_value]
                        for rule_value in probable_tail_triggered_rules
                    ]
                )

                # Get ready for printing
                # Re-create the original relation that needs to be predicted
                head_rel = id2rel[query_rel]
                # Iterate over each possible prediction and print it
                for pred_index, predicted_node in enumerate(
                    probable_tail_entities_nodes
                ):
                    # Check whether the prediction was correct
                    result = (
                        "Correct Match"
                        if predicted_node == query_tail
                        else "Wrong Match"
                    )
                    # Keep track of the body of the rule
                    rule_body = probable_tail_triggered_rules[pred_index]
                    # Keep track of the confidence of the rule
                    rule_score = probable_tail_triggered_rules_confidences[pred_index]
                    # Check whether we can deconstruct the rule to the constituent steps,
                    # If yes do so
                    # Else leave the whole rule body as a number as is
                    if math.ceil(rule_body) == rule_body:
                        # If we can factorize the rules, recreate the semantic chain
                        try:
                            chain = [
                                id2rel[prime_rel]
                                for prime_rel in get_primefactors_multiplicity(
                                    int(rule_body)
                                )
                            ]
                            # Generate the rule string
                            rule_str, first_letter, last_letter = generate_rule_string(
                                chain, head_rel
                            )
                            # Relplace the starting and ending letters with the appropriate
                            # Query values
                            rule_str = rule_str.replace(
                                f" {first_letter} ", id2node[query_head]
                            ).replace(f" {last_letter} ", id2node[query_tail])
                        except KeyError:
                            print(f"Could not decompose chain {rule_body}")
                            rule_str = f"{rule_body}({id2node[query_head]}, {id2node[query_tail]})->{id2rel[query_rel]}({id2node[query_head]}, {id2node[query_tail]}))"
                    else:
                        print(f"Could not decompose chain {rule_body}")
                        rule_str = f"{rule_body}({id2node[query_head]}, {id2node[query_tail]})->{id2rel[query_rel]}({id2node[query_head]}, {id2node[query_tail]}))"
                    # Print each prediction
                    print(f"({result}) {rule_score:.4f} : {rule_str}\t")
                    print("\n\n")
                    probable_predictions += 1
        if probable_predictions == 0:
            print(f"No rules triggered for this query...\n\n")
    return None


def predict_tail_with_ranks(
    df_test: pd.DataFrame,
    all_rules_df: pd.DataFrame,
    pam_powers: list[csr_array],
    node2id: dict[str, int],
    rel2id: dict[str, int],
    rank_rules_by_: str = "score",
    aggregate_rules_by_str: str = "max",
) -> pd.DataFrame:
    """_summary_

    Args:
        df_test (pd.DataFrame): The dataframe with columns ['head', 'rel', 'tail'] containing the test triples.
        all_rules_df (pd.DataFrame): The dataframe with the rules as generated by rule_generation.py.
        pam_powers (list[csr_array]): The PAM powers to check for triggered rules.
        node2id (dict[str, int]): The dictionary mapping nodes to their ids e.g. "Ent1"->1.
        rel2id (dict[str, int]): The dictionary mapping relations to their ids e.g. "rel1"->3
        rank_rules_by_ (str): The attribute over which we decide the importance of the rule
        aggregate_rules_by_str (str): The aggregation function. Currently either 'max' or 'noisy_or'
    Returns:
        df_results (pd.DataFrame): The predictions made. One row per test query.
    """

    if aggregate_rules_by_str == "max":
        aggregate_rules_by_func = max
    elif aggregate_rules_by_str == "noisy_or":
        aggregate_rules_by_func = noisy_or_aggregator
    else:
        raise AttributeError(
            f"Aggregate function string {aggregate_rules_by_str} not recognized.."
        )
    # Map each node to the corresponding PAM index ID
    # And each relationship to its prime number
    df_test_mapped = map_dataframe(df_test, node2id, rel2id)

    # Iterate over each query
    results = []
    for _, test_row in tqdm.tqdm(
        df_test_mapped.iterrows(), total=df_test_mapped.shape[0]
    ):

        probable_tails = []
        query_head = test_row["head"]
        query_rel = test_row["rel"]
        query_tail = test_row["tail"]

        # Keep only the rules that imply this relationship
        query_rules_all_hops = all_rules_df[all_rules_df["head_rel"] == query_rel]
        # Group them by hop-level (i.e. all rules of length 2, all rules of length 3 etc.)
        hop_level_to_rule_body_values = (
            query_rules_all_hops.groupby("hop")["body"].aggregate(list).to_dict()
        )
        # Iterate at each hop-level to find possible matches
        for k_hop_level, body_values in hop_level_to_rule_body_values.items():

            # Keep track of the confidence that each rule has
            k_hop_rules_conf = query_rules_all_hops[
                query_rules_all_hops["hop"] == k_hop_level
            ][["body", rank_rules_by_]]
            k_hop_rules_conf_dict = dict(
                zip(k_hop_rules_conf["body"], k_hop_rules_conf[rank_rules_by_])
            )
            # Find the corresponding row for this query head from the appropriate
            # k-hop matrix
            k_hop_pam_head_chains = pam_powers[k_hop_level - 1][[query_head]]
            # Find where the nnz entries of this row, match any of the rules that
            # we have recongized
            probable_tail_entities_indices = np.where(
                np.in1d(k_hop_pam_head_chains.data, body_values)
            )[0]
            # If there exist such cases, then some rules are triggered and these
            # lead us to predictions
            if probable_tail_entities_indices.shape[0] > 0:
                # Keep note of the possible tails
                probable_tail_entities_nodes = k_hop_pam_head_chains.indices[
                    probable_tail_entities_indices
                ]
                # Keep note of the rules that generated each tail
                probable_tail_triggered_rules = k_hop_pam_head_chains.data[
                    probable_tail_entities_indices
                ]
                # Keep note of the confidence of each rule
                probable_tail_triggered_rules_confidences = np.array(
                    [
                        k_hop_rules_conf_dict[rule_value]
                        for rule_value in probable_tail_triggered_rules
                    ]
                )
                probable_tail_hop_levels = np.array(
                    [k_hop_level] * len(probable_tail_entities_nodes)
                ).reshape(-1, 1)
                probable_tails.append(
                    np.hstack(
                        (
                            probable_tail_entities_nodes.reshape(-1, 1),
                            probable_tail_triggered_rules.reshape(-1, 1),
                            probable_tail_triggered_rules_confidences.reshape(-1, 1),
                            probable_tail_hop_levels,
                        )
                    )
                )
        if probable_tails:
            predictions = pd.DataFrame(
                np.vstack(probable_tails),
                columns=["Tail", "Rule_Triggered", "Rule_Confidence", "Hop_Level"],
            )
            predictions["Tail"] = predictions["Tail"].astype(int)
            sorted_predictions = (
                predictions.groupby(["Tail"])["Rule_Confidence"]
                .apply(aggregate_rules_by_func)
                .sort_values(ascending=False)
            )
            best_candidates = sorted_predictions.index.values.astype(int).tolist()
            best_candidates_scores = sorted_predictions.values.tolist()
        else:
            best_candidates = []
            best_candidates_scores = []
        test_row["predictions"] = best_candidates
        test_row["predictions_scores"] = best_candidates_scores
        try:
            test_row["rank_correct"] = best_candidates.index(query_tail) + 1
        except ValueError:
            test_row["rank_correct"] = 0
        results.append(test_row)
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":

    import time

    from data_loading import load_data
    from pam_creation import create_pam_matrices
    from rule_generation import create_ruleset
    from utils import calculate_hits_at_k

    # Data Loader details
    # Path to data folder containing train.txt and test.txt
    path = "./data/dummy_data"
    # Check load_data and change it according to the particularities of your dataset
    # e.g. the separator is '\t' instead of ','
    project_name = "test"
    # Whether to augment the graph with inverse edges
    add_inverse_edges = "NO"
    # Whether to map primes to logs. This is used for numerical overflows in
    # big data. It messes with explainability.
    use_log = False
    # What is the max k-hop PAM to create
    max_num_hops = 5
    # How far apart will the primes be
    spacing_strategy = "step_10"
    # What is the multiplication strategy
    method = "plus_times"
    # During inference, rank rules according to what attribute
    rank_rules_by_ = "score"
    # During inference, aggregate multiple rules into one by keeping their best result
    aggregate_rules_by_str = "max"

    time_s = time.time()
    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path, project_name=project_name, add_inverse_edges=add_inverse_edges, sep=","
    )
    print(f"\nLoaded Data, will create PAMs... \n")

    (
        pam_1hop_lossless,
        pam_powers,
        node2id,
        rel2id,
        broke_cause_of_sparsity,
    ) = create_pam_matrices(
        df_train,
        max_order=max_num_hops,
        method=method,
        use_log=use_log,
        spacing_strategy=spacing_strategy,
        break_with_sparsity_threshold=-1,
    )
    print(f"\nCreated PAMs, will generate rules... \n")

    all_rules_df = create_ruleset(
        pam_1hop_lossless, pam_powers, use_log=use_log, max_num_hops=-1
    )
    print(f"\nCreated {all_rules_df.shape[0]} rules, will generate predictions...  \n")

    k_hop_pams = [pam_1hop_lossless] + pam_powers[1:]

    predict_tail_with_explanations(
        df_test, all_rules_df, k_hop_pams, node2id, rel2id, rank_rules_by_="score"
    )

    df_results = predict_tail_with_ranks(
        df_test,
        all_rules_df,
        k_hop_pams,
        node2id,
        rel2id,
        rank_rules_by_="score",
        aggregate_rules_by_str="max",
    )
    _ = calculate_hits_at_k(df_results, print_=True)

    time_total = time.time() - time_s
    print(
        f"\nThe whole process took: {time_total:.0f} sec. ({time_total/60:.2f} mins).."
    )