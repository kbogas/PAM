# PAM
A bare-bones implementation of the [PAM framework](https://arxiv.org/abs/2209.06575).

Mainly used for testing.
Check each .py in its "__main__"" component where we call each of the major functions for their functionalities.

It works as it is for any collection of triples in the form of a .txt where each line corresponds to one triple
in the form:

```
ent1, rel1, ent2,
ent2, rel2, ent3
```
The delimiters can change. Please see load_data.py file for this.



## TODOS:

1. Add consistent documentation and move examples from the main sections of each .py to dedicated files or notebooks.
2. Check the effect of eliminate zeros, sort_indices in create_pam_matrices function.
3. Link prediction
   1. Add filtering cache mechanism.
   2. Parallelize (WN18RR takes about 2.8 mins)
   3. Refactor tail prediction to link prediction



