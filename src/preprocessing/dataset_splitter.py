"""
dataset_splitter.py

Utility functions for splitting the metadata into
training, validation, and testing datasets.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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


def split_dataset(
    metadata,
    train_size=0.70,
    validation_size=0.15,
    test_size=0.15,
    random_state=42
):
    """
    Splits the metadata into training,
    validation and testing datasets
    using stratified sampling.

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


def save_splits(
    train_df,
    validation_df,
    test_df,
    output_directory
):
    """
    Saves the dataset splits as CSV files.

    Parameters
    ----------
    train_df : pandas.DataFrame

    validation_df : pandas.DataFrame

    test_df : pandas.DataFrame

    output_directory : str or Path
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


def get_split_summary(
    train_df,
    validation_df,
    test_df
):
    """
    Returns the number of samples
    in each dataset split.
    """

    return {
        "Training": len(train_df),
        "Validation": len(validation_df),
        "Testing": len(test_df)
    }