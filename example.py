from prime_adj.data_loading import load_data
from prime_adj.pam_creation import create_pam_matrices
from prime_adj.rule_generation import create_ruleset
from prime_adj.tail_prediction_with_rules import predict_tail_with_explanations
from prime_adj.utils import calculate_hits_at_k

path = "./data/dummy_data/"
project_name = "test"

df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
    path, project_name=project_name, add_inverse_edges="NO", sep=","
)
print(f"\nLoaded Data, will create PAMs... \n")

(
    pam_1hop_lossless,
    pam_powers,
    node2id,
    rel2id,
    broke_with_sparsity,
) = create_pam_matrices(df_train, use_log=False, max_order=3)
print(f"\nCreated PAMs, will generate rules... \n")

all_rules_df = create_ruleset(
    pam_1hop_lossless, pam_powers, use_log=False, max_num_hops=-1
)
print(f"\nCreated {all_rules_df.shape[0]} rules, will generate predictions...  \n")

k_hop_pams = [pam_1hop_lossless] + pam_powers[1:]

predict_tail_with_explanations(
    df_test, all_rules_df, k_hop_pams, node2id, rel2id, rank_rules_by_="score"
)
