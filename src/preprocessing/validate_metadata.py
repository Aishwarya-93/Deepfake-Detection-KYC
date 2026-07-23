"""
Lab 4 - Metadata Validation

Purpose:
Validate the cleaned metadata before preprocessing
and model training.

Author: Deepfake Detection Team
"""

from pathlib import Path
import pandas as pd

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "FaceForensics++_C23"

# ==========================================================
# Load Metadata
# ==========================================================

def load_metadata():
    """
    Load the cleaned metadata.
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    print(f"Loaded {len(df)} records.")

    return df

def check_missing_values(df):
    """
    Check for missing values in the metadata.
    """

    missing = df.isnull().sum()

    print("\nMissing Values:")
    print(missing)

    return missing