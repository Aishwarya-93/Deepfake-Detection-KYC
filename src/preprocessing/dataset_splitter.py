"""
dataset_splitter.py

Utility functions for splitting the metadata into
training, validation, and testing datasets.

The FaceForensics++ "core" dataset (original + Deepfakes + FaceSwap +
Face2Face + FaceShifter + NeuralTextures) is built from 500 identity
pairs (1,000 identities, 000-999). Every manipulation method reuses the
same 500 pairings, so a naive random/stratified split on individual
videos leaks identities between train, validation and test: the same
person's face ends up in more than one split.

The functions below build an identity-pair graph, find its connected
components (each component = one identity-pair "group" of 12 videos:
2 REAL + 10 FAKE), and assign whole components to a single split so
that no core identity can ever appear in more than one split.

The DeepFakeDetection (DFD) folder is a separate actor namespace with
its own identity pairs. It is intentionally kept out of train/
validation/test entirely and treated as a held-out external/
generalization evaluation set.

Author: Deepfake Detection Team
"""

import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================================
# Namespace / identity constants
# ==========================================================

CORE_REAL_FOLDER = "original"

CORE_METHODS = [
    "Deepfakes",
    "FaceSwap",
    "Face2Face",
    "FaceShifter",
    "NeuralTextures",
]

DFD_FOLDER = "DeepFakeDetection"

CORE_REAL_PATTERN = re.compile(r"^(\d+)\.mp4$")
CORE_FAKE_PATTERN = re.compile(r"^(\d+)_(\d+)\.mp4$")
DFD_PATTERN = re.compile(r"^(\d+)_(\d+)__")

EXPECTED_CORE_GROUPS = 500
EXPECTED_CORE_VIDEOS_PER_GROUP = 12
EXPECTED_CORE_REAL_PER_GROUP = 2
EXPECTED_CORE_FAKE_PER_GROUP = 10

TRAIN_GROUPS = 350
VALIDATION_GROUPS = 75
TEST_GROUPS = 75

DEFAULT_RANDOM_STATE = 42


class SplitValidationError(Exception):
    """Raised when the identity-disjoint split fails an integrity check."""


# ==========================================================
# Metadata loading
# ==========================================================

def load_metadata(metadata_path):
    """
    Loads the master metadata CSV.

    Parameters
    ----------
    metadata_path : str or Path
        Path to master_metadata.csv.

    Returns
    -------
    pandas.DataFrame
        Loaded metadata.
    """

    return pd.read_csv(metadata_path)


# ==========================================================
# Namespace-aware identity parsing
# ==========================================================

def parse_identity(file_path):
    """
    Namespace-aware identity/group extraction for a single File Path.

    Recognises three FaceForensics++ filename shapes:

    - ``original/<id>.mp4``                 -> core, one identity
    - ``<CoreMethod>/<idA>_<idB>.mp4``       -> core, two identities
    - ``DeepFakeDetection/<a>_<b>__*.mp4``   -> dfd, two actor ids

    Parameters
    ----------
    file_path : str
        A "File Path" value from the metadata, e.g.
        "Deepfakes/000_003.mp4".

    Returns
    -------
    namespace : str
        Either "core" or "dfd".
    identities : tuple of str
        The identity id(s) referenced by this video, namespaced only
        within their own namespace (core ids and dfd ids are not
        comparable to one another).

    Raises
    ------
    ValueError
        If the File Path does not match any recognised pattern. This
        is intentional: unrecognised paths must fail loudly rather
        than silently being dropped from the identity graph.
    """

    posix_path = Path(str(file_path)).as_posix()
    parts = posix_path.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Unrecognised File Path (missing folder): {file_path!r}"
        )

    folder, filename = parts[0], parts[-1]

    if folder == CORE_REAL_FOLDER:
        match = CORE_REAL_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"Unrecognised 'original' filename: {file_path!r}"
            )
        return "core", (match.group(1),)

    if folder in CORE_METHODS:
        match = CORE_FAKE_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"Unrecognised core manipulated filename: {file_path!r}"
            )
        return "core", (match.group(1), match.group(2))

    if folder == DFD_FOLDER:
        match = DFD_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"Unrecognised DeepFakeDetection filename: {file_path!r}"
            )
        return "dfd", (match.group(1), match.group(2))

    raise ValueError(
        f"Unrecognised top-level folder {folder!r} in File Path: "
        f"{file_path!r}"
    )


# ==========================================================
# Union-Find (disjoint set) for identity-pair graphs
# ==========================================================

class UnionFind:
    """
    A minimal union-find / disjoint-set structure used to compute
    connected components of the identity-pair graph.
    """

    def __init__(self):
        self._parent = {}

    def add(self, node):
        """Registers a node, even if it is never unioned with anything."""

        self._parent.setdefault(node, node)

    def find(self, node):
        """Returns the representative (root) of the component containing node."""

        self.add(node)

        root = node
        while self._parent[root] != root:
            root = self._parent[root]

        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]

        return root

    def union(self, node_a, node_b):
        """Merges the components containing node_a and node_b."""

        root_a, root_b = self.find(node_a), self.find(node_b)

        if root_a != root_b:
            self._parent[root_a] = root_b

    def components(self):
        """
        Returns
        -------
        dict
            Mapping of an (arbitrary, non-deterministic) root node to
            the set of members in its component.
        """

        result = {}

        for node in self._parent:
            root = self.find(node)
            result.setdefault(root, set()).add(node)

        return result


# ==========================================================
# Identity-pair graph construction
# ==========================================================

def build_core_identity_groups(metadata):
    """
    Builds the core FaceForensics++ identity-pair graph and returns
    its connected components ("groups") using union-find.

    Every component is expected to contain exactly 2 identities
    (500 components total, covering the 1,000 core identities
    000-999).

    Parameters
    ----------
    metadata : pandas.DataFrame
        Full metadata (may also contain DeepFakeDetection rows; those
        are ignored here).

    Returns
    -------
    dict
        Mapping of a deterministic group id (e.g. "core_000_003") to
        a sorted tuple of the core identities in that group.
    """

    union_find = UnionFind()
    saw_core_rows = False

    for file_path in metadata["File Path"]:
        namespace, identities = parse_identity(file_path)

        if namespace != "core":
            continue

        saw_core_rows = True

        if len(identities) == 1:
            union_find.add(identities[0])
        else:
            union_find.union(identities[0], identities[1])

    if not saw_core_rows:
        raise ValueError("No core FaceForensics++ rows found in metadata.")

    groups = {}

    for members in union_find.components().values():
        member_tuple = tuple(sorted(members))
        group_id = "core_" + "_".join(member_tuple)
        groups[group_id] = member_tuple

    return groups


def build_dfd_actor_groups(metadata):
    """
    Builds the DeepFakeDetection actor-pair graph and returns its
    connected components. This is informational only (DFD is always
    held out as a single external evaluation set regardless of its
    internal component structure).

    Parameters
    ----------
    metadata : pandas.DataFrame

    Returns
    -------
    dict
        Mapping of a deterministic group id (e.g. "dfd_01_02") to a
        sorted tuple of the DFD actor ids in that group. Empty dict
        if the metadata contains no DeepFakeDetection rows.
    """

    union_find = UnionFind()
    saw_dfd_rows = False

    for file_path in metadata["File Path"]:
        namespace, identities = parse_identity(file_path)

        if namespace != "dfd":
            continue

        saw_dfd_rows = True

        if len(identities) == 1:
            union_find.add(identities[0])
        else:
            union_find.union(identities[0], identities[1])

    if not saw_dfd_rows:
        return {}

    groups = {}

    for members in union_find.components().values():
        member_tuple = tuple(sorted(members))
        group_id = "dfd_" + "_".join(member_tuple)
        groups[group_id] = member_tuple

    return groups


def annotate_identities(metadata, core_groups=None):
    """
    Returns a copy of `metadata` with additional identity/group
    columns appended (existing columns are left untouched, so any
    consumer that accesses columns by name keeps working):

    - "Identity Namespace": "core" or "dfd"
    - "Identity Ids": underscore-joined identity id(s) for that row
    - "Identity Group": core group id (None for DFD rows)

    Parameters
    ----------
    metadata : pandas.DataFrame

    core_groups : dict, optional
        Pre-computed result of `build_core_identity_groups`. Computed
        automatically if not provided.

    Returns
    -------
    pandas.DataFrame
    """

    if core_groups is None:
        core_groups = build_core_identity_groups(metadata)

    identity_to_group = {}

    for group_id, members in core_groups.items():
        for member in members:
            identity_to_group[member] = group_id

    namespaces = []
    id_strings = []
    group_ids = []

    for file_path in metadata["File Path"]:
        namespace, identities = parse_identity(file_path)

        namespaces.append(namespace)
        id_strings.append("_".join(identities))

        if namespace == "core":
            group_ids.append(identity_to_group[identities[0]])
        else:
            group_ids.append(None)

    annotated = metadata.copy()
    annotated["Identity Namespace"] = namespaces
    annotated["Identity Ids"] = id_strings
    annotated["Identity Group"] = group_ids

    return annotated


# ==========================================================
# Group -> split assignment
# ==========================================================

def assign_core_groups_to_splits(
    core_groups,
    train_groups=TRAIN_GROUPS,
    validation_groups=VALIDATION_GROUPS,
    test_groups=TEST_GROUPS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Deterministically assigns whole identity-pair groups to
    train/validation/test so that no core identity can ever appear
    in more than one split.

    Parameters
    ----------
    core_groups : dict
        Result of `build_core_identity_groups`.

    train_groups, validation_groups, test_groups : int
        Number of groups to assign to each split.

    random_state : int
        Deterministic seed used to shuffle the groups before slicing.

    Returns
    -------
    dict
        {"train": [...], "validation": [...], "test": [...]} of
        sorted group ids.
    """

    expected_total = train_groups + validation_groups + test_groups

    group_ids = sorted(core_groups.keys())

    if len(group_ids) != expected_total:
        raise ValueError(
            f"Expected {expected_total} core identity groups "
            f"({train_groups} train + {validation_groups} validation + "
            f"{test_groups} test), found {len(group_ids)}."
        )

    shuffled = group_ids[:]
    random.Random(random_state).shuffle(shuffled)

    train_ids = shuffled[:train_groups]
    validation_ids = shuffled[train_groups:train_groups + validation_groups]
    test_ids = shuffled[train_groups + validation_groups:]

    return {
        "train": sorted(train_ids),
        "validation": sorted(validation_ids),
        "test": sorted(test_ids),
    }


# ==========================================================
# Legacy (non identity-disjoint) split
# ==========================================================

def split_dataset(
    metadata,
    train_size=0.70,
    validation_size=0.15,
    test_size=0.15,
    random_state=42
):
    """
    Splits the metadata into training, validation and testing
    datasets using stratified sampling on individual videos.

    WARNING: this split is video-disjoint only. It does NOT prevent
    identity leakage — the same person's identity can (and, for
    FaceForensics++, will) appear in more than one split. Use
    `split_dataset_identity_disjoint` instead unless you specifically
    need the legacy behaviour.

    Parameters
    ----------
    metadata : pandas.DataFrame

    train_size : float

    validation_size : float

    test_size : float

    random_state : int

    Returns
    -------
    train_df
    validation_df
    test_df
    """

    if round(train_size + validation_size + test_size, 2) != 1.00:
        raise ValueError(
            "Train, validation and test sizes must add up to 1."
        )

    train_df, temp_df = train_test_split(
        metadata,
        test_size=(1 - train_size),
        random_state=random_state,
        stratify=metadata["Label"]
    )

    validation_fraction = (
        validation_size /
        (validation_size + test_size)
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - validation_fraction),
        random_state=random_state,
        stratify=temp_df["Label"]
    )

    return train_df, validation_df, test_df


# ==========================================================
# Identity-disjoint split
# ==========================================================

def split_dataset_identity_disjoint(
    metadata,
    train_groups=TRAIN_GROUPS,
    validation_groups=VALIDATION_GROUPS,
    test_groups=TEST_GROUPS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Builds an identity-disjoint split of the FaceForensics++ metadata.

    The core dataset (original + 5 manipulation methods) is split by
    whole identity-pair groups so that no core identity appears in
    more than one of train/validation/test. The DeepFakeDetection
    subset is held out entirely as a separate generalization
    evaluation set and is never placed into train/validation/test.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Full master metadata (core rows + DFD rows).

    train_groups, validation_groups, test_groups : int
        Number of core identity-pair groups to assign to each split.

    random_state : int
        Deterministic seed for the group shuffle.

    Returns
    -------
    train_df, validation_df, test_df, dfd_df : pandas.DataFrame
        Row subsets, each with the identity/group columns appended.

    report : dict
        Diagnostic information: "core_groups", "dfd_groups" and
        "assignment" (group ids per split), useful for validation
        and reporting.
    """

    core_groups = build_core_identity_groups(metadata)
    dfd_groups = build_dfd_actor_groups(metadata)

    annotated = annotate_identities(metadata, core_groups=core_groups)

    assignment = assign_core_groups_to_splits(
        core_groups,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
        random_state=random_state,
    )

    core_mask = annotated["Identity Namespace"] == "core"
    dfd_mask = annotated["Identity Namespace"] == "dfd"

    train_df = annotated[
        core_mask & annotated["Identity Group"].isin(assignment["train"])
    ].reset_index(drop=True)

    validation_df = annotated[
        core_mask & annotated["Identity Group"].isin(assignment["validation"])
    ].reset_index(drop=True)

    test_df = annotated[
        core_mask & annotated["Identity Group"].isin(assignment["test"])
    ].reset_index(drop=True)

    dfd_df = annotated[dfd_mask].reset_index(drop=True)

    report = {
        "core_groups": core_groups,
        "dfd_groups": dfd_groups,
        "assignment": assignment,
    }

    return train_df, validation_df, test_df, dfd_df, report


# ==========================================================
# Validation checks (fail loudly)
# ==========================================================

def validate_identity_disjoint_split(
    train_df,
    validation_df,
    test_df,
    dfd_df,
    expected_train_groups=TRAIN_GROUPS,
    expected_validation_groups=VALIDATION_GROUPS,
    expected_test_groups=TEST_GROUPS,
):
    """
    Runs strong integrity checks against an identity-disjoint split
    and raises `SplitValidationError` loudly if any check fails.

    Checks performed:

    1. No "File Path" occurs in more than one split (including DFD).
    2. No core identity occurs in more than one of train/val/test.
    3. No core group occurs in more than one of train/val/test.
    4. Every expected core group is assigned exactly once (no
       missing, no duplicated groups).
    5. No DFD row appears inside train/validation/test.
    6. The dedicated DFD split contains only DFD rows.
    7. Each split has the expected number of core groups.
    8. Each split has the expected number of videos and REAL/FAKE
       counts, derived from the expected group counts (every core
       group contributes exactly 2 REAL + 10 FAKE videos).

    Parameters
    ----------
    train_df, validation_df, test_df, dfd_df : pandas.DataFrame
        Output of `split_dataset_identity_disjoint`. Must contain the
        "Identity Namespace", "Identity Ids" and "Identity Group"
        columns produced by `annotate_identities`.

    expected_train_groups, expected_validation_groups,
    expected_test_groups : int

    Returns
    -------
    bool
        True if every check passed.

    Raises
    ------
    SplitValidationError
        If any check fails. The exception message lists every
        violation found (not just the first).
    """

    errors = []

    core_splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    all_splits = dict(core_splits)
    all_splits["deepfakedetection"] = dfd_df

    # 1. File Path uniqueness across every split.
    path_to_splits = {}
    for split_name, df in all_splits.items():
        for path in df["File Path"]:
            path_to_splits.setdefault(path, set()).add(split_name)

    duplicated_paths = {
        path: splits for path, splits in path_to_splits.items()
        if len(splits) > 1
    }
    if duplicated_paths:
        sample = list(duplicated_paths.items())[:5]
        errors.append(
            f"{len(duplicated_paths)} File Path value(s) appear in more "
            f"than one split, e.g. {sample}"
        )

    # 2. Core identity uniqueness across train/validation/test only
    #    (DFD identities live in a disjoint namespace by construction).
    identity_to_splits = {}
    for split_name, df in core_splits.items():
        for id_string in df["Identity Ids"]:
            for identity in id_string.split("_"):
                identity_to_splits.setdefault(identity, set()).add(split_name)

    leaked_identities = {
        identity: splits for identity, splits in identity_to_splits.items()
        if len(splits) > 1
    }
    if leaked_identities:
        sample = list(leaked_identities.items())[:5]
        errors.append(
            f"{len(leaked_identities)} core identity(ies) leak across "
            f"splits, e.g. {sample}"
        )

    # 3. Core group uniqueness across train/validation/test.
    group_to_splits = {}
    for split_name, df in core_splits.items():
        for group_id in df["Identity Group"].dropna().unique():
            group_to_splits.setdefault(group_id, set()).add(split_name)

    leaked_groups = {
        group_id: splits for group_id, splits in group_to_splits.items()
        if len(splits) > 1
    }
    if leaked_groups:
        errors.append(
            f"{len(leaked_groups)} core identity group(s) leak across "
            f"splits: {leaked_groups}"
        )

    # 4. Every expected group assigned exactly once (no missing / duplicated).
    expected_total_groups = (
        expected_train_groups + expected_validation_groups + expected_test_groups
    )
    all_assigned_groups = set(group_to_splits.keys())

    if len(all_assigned_groups) != expected_total_groups:
        errors.append(
            f"Expected {expected_total_groups} unique core identity "
            f"groups across all splits, found {len(all_assigned_groups)} "
            f"(missing or duplicated groups)."
        )

    # 5 & 6. DFD isolation.
    for split_name, df in core_splits.items():
        if (df["Identity Namespace"] == "dfd").any():
            errors.append(
                f"DFD rows were found inside the '{split_name}' split."
            )

    if len(dfd_df) == 0:
        errors.append("The DeepFakeDetection split is empty.")
    elif (dfd_df["Identity Namespace"] != "dfd").any():
        errors.append(
            "The DeepFakeDetection split contains non-DFD rows."
        )

    # 7. Expected group counts per split.
    expected_group_counts = {
        "train": expected_train_groups,
        "validation": expected_validation_groups,
        "test": expected_test_groups,
    }
    for split_name, expected in expected_group_counts.items():
        actual = core_splits[split_name]["Identity Group"].nunique()
        if actual != expected:
            errors.append(
                f"Split '{split_name}' has {actual} identity groups, "
                f"expected {expected}."
            )

    # 8. Expected video / REAL / FAKE counts per split (derived from
    #    group counts: every core group = 2 REAL + 10 FAKE videos).
    for split_name, expected_groups in expected_group_counts.items():
        df = core_splits[split_name]
        expected_videos = expected_groups * EXPECTED_CORE_VIDEOS_PER_GROUP
        expected_real = expected_groups * EXPECTED_CORE_REAL_PER_GROUP
        expected_fake = expected_groups * EXPECTED_CORE_FAKE_PER_GROUP

        actual_videos = len(df)
        actual_real = int((df["Label"] == "REAL").sum())
        actual_fake = int((df["Label"] == "FAKE").sum())

        if actual_videos != expected_videos:
            errors.append(
                f"Split '{split_name}' has {actual_videos} videos, "
                f"expected {expected_videos}."
            )
        if actual_real != expected_real:
            errors.append(
                f"Split '{split_name}' has {actual_real} REAL videos, "
                f"expected {expected_real}."
            )
        if actual_fake != expected_fake:
            errors.append(
                f"Split '{split_name}' has {actual_fake} FAKE videos, "
                f"expected {expected_fake}."
            )

    if errors:
        raise SplitValidationError(
            "Identity-disjoint split validation FAILED:\n- "
            + "\n- ".join(errors)
        )

    return True


# ==========================================================
# Reporting helpers
# ==========================================================

def compute_identity_overlap_matrix(train_df, validation_df, test_df):
    """
    Computes a 3x3 matrix of core-identity overlap counts between
    train/validation/test. Every off-diagonal entry must be 0 for a
    valid identity-disjoint split.

    Returns
    -------
    pandas.DataFrame
        Index/columns are ["train", "validation", "test"].
    """

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    identity_sets = {}
    for split_name, df in splits.items():
        identities = set()
        for id_string in df["Identity Ids"]:
            identities.update(id_string.split("_"))
        identity_sets[split_name] = identities

    names = list(splits.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype=int)

    for row_name in names:
        for col_name in names:
            matrix.loc[row_name, col_name] = len(
                identity_sets[row_name] & identity_sets[col_name]
            )

    return matrix


def summarize_identity_split(train_df, validation_df, test_df, dfd_df):
    """
    Builds a dictionary of the diagnostics required to demonstrate
    and verify the identity-disjoint split: videos per split,
    REAL/FAKE counts, unique identities per split, the identity
    overlap matrix, groups per split, and the DFD held-out count.
    """

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    return {
        "videos_per_split": {
            name: len(df) for name, df in splits.items()
        },
        "label_counts_per_split": {
            name: df["Label"].value_counts().to_dict()
            for name, df in splits.items()
        },
        "unique_identities_per_split": {
            name: len({
                identity
                for id_string in df["Identity Ids"]
                for identity in id_string.split("_")
            })
            for name, df in splits.items()
        },
        "groups_per_split": {
            name: int(df["Identity Group"].nunique())
            for name, df in splits.items()
        },
        "identity_overlap_matrix": compute_identity_overlap_matrix(
            train_df, validation_df, test_df
        ),
        "dfd_video_count": len(dfd_df),
        "dfd_label_counts": dfd_df["Label"].value_counts().to_dict(),
    }


# ==========================================================
# Saving splits
# ==========================================================

def save_splits(
    train_df,
    validation_df,
    test_df,
    output_directory,
    dfd_df=None
):
    """
    Saves the dataset splits as CSV files.

    Parameters
    ----------
    train_df : pandas.DataFrame

    validation_df : pandas.DataFrame

    test_df : pandas.DataFrame

    output_directory : str or Path

    dfd_df : pandas.DataFrame, optional
        DeepFakeDetection held-out rows. If provided, saved as
        "deepfakedetection_test.csv" alongside the core splits.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        output_directory / "train.csv",
        index=False
    )

    validation_df.to_csv(
        output_directory / "validation.csv",
        index=False
    )

    test_df.to_csv(
        output_directory / "test.csv",
        index=False
    )

    if dfd_df is not None:
        dfd_df.to_csv(
            output_directory / "deepfakedetection_test.csv",
            index=False
        )


def get_split_summary(
    train_df,
    validation_df,
    test_df,
    dfd_df=None
):
    """
    Returns the number of samples in each dataset split.
    """

    summary = {
        "Training": len(train_df),
        "Validation": len(validation_df),
        "Testing": len(test_df)
    }

    if dfd_df is not None:
        summary["DeepFakeDetection"] = len(dfd_df)

    return summary


# ==========================================================
# CLI entry point
# ==========================================================

def main():
    """
    Regenerates data/splits/{train,validation,test,
    deepfakedetection_test}.csv from data/metadata/master_metadata.csv
    using the identity-disjoint split, validating the result before
    writing anything to disk.
    """

    metadata_path = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"
    output_dir = PROJECT_ROOT / "data" / "splits"

    print(f"Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)

    print("Building identity-disjoint split (seed=42)...")
    train_df, validation_df, test_df, dfd_df, report = (
        split_dataset_identity_disjoint(metadata, random_state=DEFAULT_RANDOM_STATE)
    )

    print("Validating split integrity...")
    validate_identity_disjoint_split(train_df, validation_df, test_df, dfd_df)
    print("All validation checks passed.")

    save_splits(train_df, validation_df, test_df, output_dir, dfd_df=dfd_df)

    summary = get_split_summary(train_df, validation_df, test_df, dfd_df=dfd_df)

    print("\nIdentity-disjoint split written successfully:")
    for split_name, count in summary.items():
        print(f"  {split_name}: {count}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
