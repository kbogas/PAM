import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from grakel.datasets import fetch_dataset
from grakel.datasets.base import dataset_metadata
from grakel.kernels import (
    EdgeHistogram,
    VertexHistogram,
    WeisfeilerLehman,
    WeisfeilerLehmanOptimalAssignment,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from grakel_utils import ProductPower
from utils import get_prime_adjacency

timestamp = "__" + datetime.now().strftime("%H:%M-%d_%m_%Y")


##### SETTINGS ####
log_filename = "test_gridsearch"
results_path_csv = "./results/gridsearch_graph_classification.csv"
num_folds_outer_cv = 5
num_folds_inner_cv = 3

#####################


def setup_logger():
    """
    Setting up logging.
    """

    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("../logs", log_filename + timestamp)),
            logging.StreamHandler(),
        ],
    )
    logging.info("Created the log.file")
    logging.info(f"Starting experiment : {timestamp[2:]}")
    logging.info("~" * 30)
    return


setup_logger()


# We want edge labels as these are the multi-relational
# We need node labels for the other methodologies
# Tox datasets seem to have problems
wanted_dataset_names = []
for dataset_name, values in dataset_metadata.items():
    if values["el"] and values["nl"] and not ("Tox" in dataset_name):
        wanted_dataset_names.append(dataset_name)

# We remove some problematic files
# Tox21_AHR has an empty graph,
# Tox21_AR is wronly packaged (the downloaded file after unzipping is named something else and throws error
# Cuneiform soemthing wrong with the node labels being strings instead of ints?
# Zinc and Alchemy do not have graph labels. Maybe need to download train,test,valid
unwanted = set(
    [
        "Cuneiform",
        "Tox21_AHR",
        "Tox21_AR",
        "ZINC_full",
        "alchemy_full",
    ]
)
wanted_dataset_names = [dt for dt in wanted_dataset_names if not (dt in unwanted)]

# wanted_dataset_names = extra_wanted
logging.info(f"Total: {len(wanted_dataset_names)} datasets")
for dt in wanted_dataset_names:
    logging.info(dt)


# Define the models
models = [
    VertexHistogram(normalize=False),
    EdgeHistogram(normalize=False),
    WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=True),
    WeisfeilerLehmanOptimalAssignment(n_jobs=-1, n_iter=3),
    ProductPower(
        power=1,
        aggr_str="log",
        use_ohe=False,
        normalize=False,
        use_laplace=False,
        grakel_compatible=True,
        kernel_str="rbf",
    ),
    ProductPower(
        power=1,
        aggr_str="log",
        use_ohe=False,
        normalize=False,
        use_laplace=False,
        grakel_compatible=True,
        kernel_str="rbf",
    ),
]

model_names = [
    "VH",
    "EH",
    "WL-1",
    "WL-OA",
    "PP-1",
    "PP-1+VH",
]


# Outer cv for validation
cv_outer = StratifiedKFold(random_state=42, n_splits=num_folds_outer_cv, shuffle=True)
# Inner cv for hyper-param tuning
cv_inner = StratifiedKFold(random_state=42, n_splits=num_folds_inner_cv, shuffle=True)

# Aggregate all results here
results = []
for dataset_name in wanted_dataset_names[:]:
    logging.info(f"Dataset: {dataset_name}")

    dataset = fetch_dataset(dataset_name, verbose=False, download_if_missing=True)

    # Load the graphs
    G, y = dataset.data, dataset.target
    logging.info(f"Num graphs: {len(G)}")

    # Create the corresponding adjacency matrices
    prime_adj_all = np.array(
        [
            get_prime_adjacency(
                g[2],
                nodes_dict=None,
                add_inverse_edges=False,
                starting_value=1,
                rel_spacing_strategy="factor_5",
            )
            for g in G
        ],
        dtype="object",
    )

    logging.info(f"Parsed {dataset_name}")
    # Splits the dataset into a training and a test set
    for fold_index, (train_indices, test_indices) in enumerate(cv_outer.split(G, y)):
        for model, model_name in zip(models, model_names):
            logging.info(f"Model: {model_name}")

            time_s = time.time()

            # We need to fit using the PAM-based adjacencies
            if "PP" in model_name:
                X = prime_adj_all
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]
                K_train = model.fit_transform(X_train, y_train)
                K_test = model.transform(X_test)
            else:
                # We need to fit using the original adjacencies#\
                X = np.array(G)
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]
                K_train = model.fit_transform(X_train)
                K_test = model.transform(X_test)
            # Just pass-through here keeping track of the kernels
            # for the VH and the time it took
            if model_name == "VH":
                K_train_VH = K_train.copy()
                K_test_VH = K_test.copy()
                time_took_tr_VH = time.time() - time_s

            # Add the VH Gramm matrix to the PAM-based one
            elif model_name == "PP-1+VH":
                K_train = 0.5 * K_train + 0.5 * K_train_VH
                K_test = 0.5 * K_test + 0.5 * K_test_VH

            # Load the same classifier for all
            clf = SVC(kernel="precomputed")
            pipe = Pipeline(
                [
                    ("clf", clf),
                ]
            )

            # Same Grid-search for all kernels
            param_grid = {
                "clf__C": [10**p for p in [-3, -2, -1, 0, 1, 2, 3]],
            }

            grid = GridSearchCV(
                pipe,
                param_grid,
                refit=True,
                cv=cv_inner,
                verbose=1,
                n_jobs=-1,
                scoring="f1_micro",
            )
            grid.fit(K_train, y_train)
            # Report metrics
            y_pred = grid.predict(K_test)
            acc = accuracy_score(y_test, y_pred)
            f1_micro = f1_score(y_test, y_pred, average="micro")
            f1_macro = f1_score(y_test, y_pred, average="macro")
            time_took = time.time() - time_s
            if model_name == "PP-1+VH":
                time_took += time_took_tr_VH
            results.append(
                {
                    "Dataset": dataset_name,
                    "Fold": fold_index,
                    "Model": model_name,
                    "Acc": acc,
                    "F1_macro": f1_macro,
                    "F1_micro": f1_micro,
                    "Time": time_took,
                    **grid.best_params_,
                }
            )
            logging.info(results[-1])
    m = pd.DataFrame(results)
    m_cur = (
        m[m["Dataset"] == dataset_name]
        .groupby("Model")[["Acc", "F1_macro", "F1_micro", "Time"]]
        .mean()
        .sort_values("f1_micro", ascending=False)
    )
    logging.info(m_cur.to_string())
    logging.info("~" * 20)


# Final metric to score against
score_metric = "F1_micro"


# Save results and print wins and losses
res = pd.DataFrame(results)
logging.info(
    res.groupby(["Dataset", "Model"])[["F1_micro", "F1_macro", "Acc", "Time"]]
    .mean()
    .sort_values(score_metric, ascending=False)
)
res.to_csv(results_path_csv, index=False)


model_names = res.Model.unique().tolist()
wanted_dataset_names = res["Dataset"].unique().tolist()

# Also print Time wins and Score wins between the models
wins_time = np.zeros((len(model_names), len(model_names)))
wins_score = np.zeros((len(model_names), len(model_names)))
for dataset in wanted_dataset_names:
    subset = (
        res[res["Dataset"] == dataset].groupby("Model")[[score_metric, "Time"]].mean()
    )
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names[:]):
            if subset.loc[m1][score_metric] > subset.loc[m2][score_metric]:
                wins_score[i, j] += 1

            if subset.loc[m1]["Time"] < subset.loc[m2]["Time"]:
                wins_time[i, j] += 1


df_time = pd.DataFrame(wins_time, index=model_names, columns=model_names)
df_score = pd.DataFrame(wins_score, index=model_names, columns=model_names)
logging.info("\n")
logging.info(f"Time Wins")
logging.info(df_time)
logging.info("\n")
logging.info("Score Wins")
logging.info(df_score)
