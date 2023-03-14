import os

import pandas as pd
from sympy import nextprime, primefactors


def load_csv(
    path_to_file: str,
    add_inverse_edges: str = "NO",
    pandas_sep=",",
    pandas_first_row_header=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function that loads the data in pd.DataFrames and returns them.
    Args:
        path_to_file (str): path to folder with train.txt, valid.txt, test.txt
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".
        pandas_sep (str, optional): the separator to use in pd.read_csv,
        when loading the data. Defaults to ",".
        pandas_first_row_header (bool, optional): whether the first row of the file
        contains a header for the data. Defaults to False.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the triples loaded in a pandas. We return
        both the original triple-set, as well as the (possibly) augmented one
        (i.e. with inverse edges)
    """

    if pandas_first_row_header:
        header_row = 0
    else:
        header_row = None

    df_train = pd.read_csv(
        path_to_file,
        sep=pandas_sep,
        header=header_row,
        dtype="str",
    )
    df_train.columns = ["head", "rel", "tail"]  # type: ignore
    df_train_orig = df_train.copy()
    if "YES" in add_inverse_edges:
        print(f"Will add the inverse train edges as well..")
        df_train["rel"] = df_train["rel"].astype(str)
        df_train_inv = df_train.copy()
        df_train_inv["head"] = df_train["tail"]
        df_train_inv["tail"] = df_train["head"]
        if add_inverse_edges == "YES__INV":
            df_train_inv["rel"] = df_train["rel"] + "__INV"
        df_train = pd.concat((df_train, df_train_inv))
    return df_train_orig, df_train


def load_whole_dataset(
    path_to_folder: str, project_name: str, add_inverse_edges: str = "NO"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Helper function that loads a whole dataset in pd.DataFrames and returns them.
    Args:
        path_to_folder (str): path to folder with train.txt, valid.txt, test.txt
        project_name (str): name of the project
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    PROJECT_DETAILS = {
        "lc-neo4j": {"skiprows": 1, "sep": "\t"},
        "codex-s": {"skiprows": 0, "sep": "\t"},
        "WN18RR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10-DR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10": {"skiprows": 0, "sep": "\t"},
        "FB15k-237": {"skiprows": 0, "sep": "\t"},
        "NELL995": {"skiprows": 0, "sep": "\t"},
        "DDB14": {"skiprows": 0, "sep": "\t"},
    }

    df_train = pd.read_csv(
        os.path.join(path_to_folder, "train.txt"),
        sep=PROJECT_DETAILS[project_name]["sep"],
        header=None,
        dtype="str",
        skiprows=PROJECT_DETAILS[project_name]["skiprows"],
    )
    df_eval = []
    df_test = []
    df_train.columns = ["head", "rel", "tail"]  # type: ignore
    df_train_orig = df_train.copy()
    if "YES" in add_inverse_edges:
        print(f"Will add the inverse train edges as well..")
        df_train["rel"] = df_train["rel"].astype(str)
        df_train_inv = df_train.copy()
        df_train_inv["head"] = df_train["tail"]
        df_train_inv["tail"] = df_train["head"]
        if add_inverse_edges == "YES__INV":
            df_train_inv["rel"] = df_train["rel"] + "__INV"
        df_train = pd.concat((df_train, df_train_inv))
    if project_name in ["lc-neo4j"]:
        df_eval = pd.DataFrame()
        df_test = pd.DataFrame()
        already_seen_triples = set(df_train.to_records(index=False).tolist())
    else:
        try:
            df_eval = pd.read_csv(
                os.path.join(path_to_folder, "valid.txt"),
                sep=PROJECT_DETAILS[project_name]["sep"],
                header=None,
                dtype="str",
                skiprows=PROJECT_DETAILS[project_name]["skiprows"],
            )
            df_eval.columns = ["head", "rel", "tail"]  # type: ignore
        except FileNotFoundError:
            print(
                f"No valid.txt found in {path_to_folder}... df_eval will contain the train data.."
            )
            df_eval = df_train.copy()
        df_test = pd.read_csv(
            os.path.join(path_to_folder, "test.txt"),
            sep=PROJECT_DETAILS[project_name]["sep"],
            header=None,
            dtype="str",
            skiprows=PROJECT_DETAILS[project_name]["skiprows"],
        )
        df_test.columns = ["head", "rel", "tail"]  # type: ignore
        if "YAGO" in project_name:
            for cur_df in [df_train, df_eval, df_test]:
                for col in cur_df.columns:
                    cur_df[col] = cur_df[col]  # + "_YAGO"

        already_seen_triples = set(
            df_train.to_records(index=False).tolist()
            + df_eval.to_records(index=False).tolist()
        )
    print(f"Total: {len(already_seen_triples)} triples in train + eval!)")
    print(f"In train: {len(df_train)}")
    print(f"In valid: {len(df_eval)}")
    print(f"In test: {len(df_test)}")
    return df_train_orig, df_train, df_eval, df_test, already_seen_triples


if __name__ == "__main__":
    path = "../test/dummy_data/train.txt"
    df_train_orig, df_train = load_csv(path, add_inverse_edges="YES")
    print(df_train_orig)
    assert df_train_orig.shape[0] * 2 == df_train.shape[0]
