# WN18RR relation-prediction walkthrough

The authors' PAM/BoP experiment repository reports this WN18RR split:

| Split | Triples |
| --- | ---: |
| Train | 86,835 |
| Validation | 3,034 |
| Test | 3,134 |

It contains 40,559 nodes and 11 relation types. Load the directory with
`load_data`, fit the relation map and BoP vocabulary on the training graph and
training query objects only, use validation for model selection, then evaluate
once on held-out test queries. Exclude target triples and their inverses from
query evidence.

For each query, rank candidate relations or entities by descending score.
Mean reciprocal rank is the mean of `1/rank` for the correct answer. Hits@K is
the fraction whose correct answer is ranked at most K. Filtered evaluation
removes other known correct answers while retaining the query answer.

The saved reference output in `experiments/results/wn18rr_reference.csv` records
the authors' reported lossy and lossless scores, feature dimensions, and CPU
times. Those values are provenance records, not CI performance thresholds.

Source: [PAM/BoP relation-prediction experiments](https://github.com/kbogas/PAM_BoP/tree/main/relation_prediction).
