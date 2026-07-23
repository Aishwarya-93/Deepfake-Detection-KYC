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


# ==========================================================
# Missing Values
# ==========================================================

def check_missing_values(df):
    """
    Check for missing values in the metadata.
    """

    missing = df.isnull().sum()

    print("\n========== Missing Values ==========")
    print(missing)

    return missing


# ==========================================================
# Duplicate Rows
# ==========================================================

def check_duplicates(df):
    """
    Check for duplicate rows.
    """

    duplicates = df.duplicated().sum()

    print("\n========== Duplicate Rows ==========")
    print(f"Duplicate Rows: {duplicates}")

    return duplicates


# ==========================================================
# Class Distribution
# ==========================================================

def check_class_distribution(df):
    """
    Display class distribution.
    """

    distribution = df["Label"].value_counts()

    print("\n========== Class Distribution ==========")
    print(distribution)

    return distribution


# ==========================================================
# Frame Statistics
# ==========================================================

def frame_statistics(df):
    """
    Display statistics for the Frame Count column.
    """

    print("\n========== Frame Statistics ==========")

    statistics = df["Frame Count"].describe()

    print(statistics)

    return statistics


# ==========================================================
# Resolution Statistics
# ==========================================================

def resolution_statistics(df):
    """
    Display video resolution statistics.
    """

    print("\n========== Resolution Statistics ==========")

    resolution = (
        df["Width"].astype(str)
        + " x "
        + df["Height"].astype(str)
    )

    counts = resolution.value_counts()

    print(counts)

    return counts


# ==========================================================
# Video Path Validation
# ==========================================================

def check_video_paths(df):
    """
    Verify that every video listed in the metadata exists.
    """

    print("\n========== Video Path Validation ==========")

    missing_files = []

    for file_path in df["File Path"]:

        video_path = DATASET_PATH / file_path

        if not video_path.exists():
            missing_files.append(file_path)

    print(f"Missing files : {len(missing_files)}")

    if missing_files:
        print("\nFirst 10 missing files:")

        for file in missing_files[:10]:
            print(file)

    return missing_files


# ==========================================================
# Validation Report
# ==========================================================

def generate_validation_report(df):
    """
    Generate a complete metadata validation report.
    """

    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    class_distribution = df["Label"].value_counts()

    frame_stats = frame_statistics(df)
    resolution_counts = resolution_statistics(df)
    missing_files = check_video_paths(df)

    status = "PASS"

    if (
        missing_values > 0
        or duplicate_rows > 0
        or len(missing_files) > 0
    ):
        status = "FAIL"

    print("\n===================================")
    print("     Metadata Validation Report")
    print("===================================")

    print(f"Total Videos         : {len(df)}")
    print(f"Missing Values       : {missing_values}")
    print(f"Duplicate Rows       : {duplicate_rows}")
    print(f"Missing Files        : {len(missing_files)}")
    print(f"Average Frames       : {frame_stats['mean']:.2f}")
    print(f"Resolution Types     : {len(resolution_counts)}")

    print("\nClass Distribution:")

    for label, count in class_distribution.items():
        print(f"{label:5}: {count}")

    print(f"\nValidation Status : {status}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    metadata = load_metadata()

    generate_validation_report(metadata)