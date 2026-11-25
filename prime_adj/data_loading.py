import os
import warnings

import pandas as pd


TRIPLE_COLUMNS = ["head", "rel", "tail"]
INVERSE_MODES = {"NO", "YES", "YES__INV"}


def _empty_triples() -> pd.DataFrame:
    return pd.DataFrame(columns=TRIPLE_COLUMNS, dtype="str")


def _augment_inverse_edges(
    triples: pd.DataFrame, add_inverse_edges: str
) -> pd.DataFrame:
    """Return a fresh training graph with the requested inverse augmentation."""

    if add_inverse_edges not in INVERSE_MODES:
        raise ValueError(
            f"add_inverse_edges must be one of {sorted(INVERSE_MODES)}, "
            f"got {add_inverse_edges!r}"
        )
    augmented = triples.copy(deep=True).reset_index(drop=True)
    if add_inverse_edges == "NO":
        return augmented
    inverse = augmented.rename(columns={"head": "tail", "tail": "head"})[
        TRIPLE_COLUMNS
    ].copy()
    if add_inverse_edges == "YES__INV":
        inverse["rel"] = inverse["rel"].astype(str) + "__INV"
    return pd.concat([augmented, inverse], ignore_index=True)


def _read_triples(path: str, *, sep: str, skiprows: int) -> pd.DataFrame:
    triples = pd.read_csv(
        path, sep=sep, header=None, dtype="str", skiprows=skiprows
    )
    if triples.shape[1] != 3:
        raise ValueError(f"Expected three columns in {path}, found {triples.shape[1]}")
    triples.columns = TRIPLE_COLUMNS
    return triples


def require_held_out_split(
    split: pd.DataFrame, split_name: str = "test"
) -> pd.DataFrame:
    """Reject an absent evaluation split before metrics are computed."""

    if split.empty:
        raise ValueError(
            f"A non-empty held-out {split_name} split is required for evaluation"
        )
    if list(split.columns) != TRIPLE_COLUMNS:
        raise ValueError(
            f"Held-out {split_name} split must have columns {TRIPLE_COLUMNS}"
        )
    return split


def load_csv(
    path_to_file: str,
    add_inverse_edges: str = "NO",
    pandas_sep=",",
    pandas_first_row_header=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function that loads the collection of triples in pd.DataFrames and returns them.
    It expects a collection of triples as .csv where each triple is in one line. E.g.
    entity1, relation1, entity2,
    entity1, relation2, entity3,
    ...
    If there is a header first row, please pass pandas_first_row_header=True
    Args:
        path_to_file (str): path to .csv co
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".
        If "YES" will simply add the inverse edges. Eg.
        If "entity1, relation1, entity2" exists it will add "entity2, relation1, entity1"
        If "YES__INV" it will add the inverse edges with a suffix. Eg.
        If "entity1, relation1, entity2" exists it will add "entity2, relation1__INV, entity1"
        If "NO" it will add no new triples.
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
    df_train.columns = TRIPLE_COLUMNS  # type: ignore
    df_train_orig = df_train.copy()
    df_train = _augment_inverse_edges(df_train, add_inverse_edges)
    return df_train_orig, df_train


def load_data(
    path_to_folder: str,
    project_name: str,
    add_inverse_edges: str = "NO",
    skiprows: int = 0,
    sep: str = "\t",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Helper function that loads the data in pd.DataFrames and returns them.
    No header row is expected here. It follows the same convention as load_csv dataset.
    Args:
        path_to_folder (str): path to folder with train.txt, valid.txt, test.txt
        project_name (str): name of the project
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".
        skiprows (int, optional): Number of rows to skip. Pleas provide if new dataset.
        Default is 0.
        sep (str, optional): Separator for read_csv. Please provide if new dataset.
        Default is '\t'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]: [description]
    """
    PROJECT_DETAILS = {
        "lc-neo4j": {"skiprows": 1, "sep": "\t"},
        "ad-neo4j": {"skiprows": 1, "sep": ","},
        "codex-s": {"skiprows": 0, "sep": "\t"},
        "codex-l": {"skiprows": 0, "sep": "\t"},
        "WN18RR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10-DR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10": {"skiprows": 0, "sep": "\t"},
        "FB15k-237": {"skiprows": 0, "sep": "\t"},
        "NELL995": {"skiprows": 0, "sep": "\t"},
        "DDB14": {"skiprows": 0, "sep": "\t"},
    }

    # Try out with generic skiprows=0 and sep='\t'
    if not project_name in PROJECT_DETAILS:
        PROJECT_DETAILS[project_name] = {"skiprows": skiprows, "sep": sep}

    split_sep = PROJECT_DETAILS[project_name]["sep"]
    split_skiprows = PROJECT_DETAILS[project_name]["skiprows"]
    df_train_orig = _read_triples(
        os.path.join(path_to_folder, "train.txt"),
        sep=split_sep,
        skiprows=split_skiprows,
    )
    df_train = _augment_inverse_edges(df_train_orig, add_inverse_edges)

    def optional_split(filename: str, split_name: str) -> pd.DataFrame:
        try:
            return _read_triples(
                os.path.join(path_to_folder, filename),
                sep=split_sep,
                skiprows=split_skiprows,
            )
        except FileNotFoundError:
            warnings.warn(
                f"No {filename} found in {path_to_folder}; {split_name} is empty. "
                "Training triples are not substituted for held-out data.",
                UserWarning,
                stacklevel=2,
            )
            return _empty_triples()

    df_eval = optional_split("valid.txt", "validation")
    df_test = optional_split("test.txt", "test")
    already_seen_triples = set(
        df_train.to_records(index=False).tolist()
        + df_eval.to_records(index=False).tolist()
    )
    print(f"Total: {len(already_seen_triples)} triples in train + eval!)")
    print(f"In train: {len(df_train)}")
    print(f"In valid: {len(df_eval)}")
    print(f"In test: {len(df_test)}")
    return df_train_orig, df_train, df_eval, df_test, already_seen_triples
