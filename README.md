# Prime Adjacency Matrix
A bare-bones implementation of the [PAM framework](https://arxiv.org/abs/2209.06575), i.e. Prime Adjacency Matrices for Multi-relational Networks.


Mainly used for testing.
Check each .py in its "__main__"" component where we call each of the major functions for their functionalities.

## Installation
You can install it manually (by cloining and using requiremets.txt) or through pip:

```cmd
pip install prime_adj
```

## Usage

It works using as input any collection of triples in the form of a .txt where each line corresponds to one triple
in the form:

```
ent1, rel1, ent2,
ent2, rel2, ent3
```
The delimiters can change. Please see load_data.py file for this.

You can check each of the .py files for the specific functionality you are interested in.
For example, running:

```py

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

```

Will load the dummy data, calculate up to 3-hop PAMs, generate rules, predict possible tail candidates for each the two test queries in the *./test/dummy_data/test.txt* file and print those predictions...

Specifically, you will get:

```cmd
No valid.txt found in ../test/dummy_data... df_eval will contain the train data..
Total: 9 triples in train + eval!)
In train: 9
In valid: 9
In test: 2

Loaded Data, will create PAMs... 

# of unique rels: 3      | # of unique nodes: 5
(5, 5) Sparsity: 68.00 %
Hop 2
Sparsity 2-hop: 76.00 %
Hop 3
Sparsity 3-hop: 88.00 %
Hop 4
Sparsity 4-hop: 96.00 %
Hop 5
Sparsity 5-hop: 100.00 %

Created PAMs, will generate rules... 

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 806.19it/s]

Created 12 rules, will generate predictions...  

Query (0): Nick - knows - Anna
(Correct Match) 0.4000 : knows(Nick, B ) ^ knows( B ,Anna) -> knows(Nick,Anna)



Query (1): George - lives_in - Athens
(Correct Match) 0.5000 : works_in(George,Athens) -> lives_in(George,Athens)



100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 300.44it/s]
Tail Hits@1: 100.00 %
Tail Hits@3: 100.00 %
Tail Hits@5: 100.00 %
Tail Hits@10: 100.00 %
Tail Hits@UpperBound: 100.00 %
```



## TODOS:

1. Add consistent documentation and move examples from the main sections of each .py to dedicated files or notebooks.
2. Check the effect of eliminate zeros, sort_indices in create_pam_matrices function.
3. Link prediction
   1. Add filtering cache mechanism.
   2. Parallelize (WN18RR takes about 2.8 mins)
   3. Refactor tail prediction to link prediction



