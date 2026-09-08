"""
Lab 3 - Metadata Engineering

Purpose:
Read all FaceForensics++ metadata CSV files,
merge them into one master metadata file,
clean unnecessary columns,
and save the result.

Author: Deepfake Detection Team
"""

from pathlib import Path
import pandas as pd

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "FaceForensics++_C23"

CSV_FOLDER = DATASET_PATH / "csv"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "metadata"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
# ==========================================================
# CSV Files
# ==========================================================

CSV_FILES = [
    "original.csv",
    "Deepfakes.csv",
    "FaceSwap.csv",
    "Face2Face.csv",
    "FaceShifter.csv",
    "NeuralTextures.csv",
    "DeepFakeDetection.csv"
]

# ==========================================================
# Read Metadata
# ==========================================================

def load_csv_files():
    """
    Read all required metadata CSV files
    and merge them into a single DataFrame.
    """

    dataframes = []

    for file_name in CSV_FILES:

        file_path = CSV_FOLDER / file_name

        print(f"Reading {file_name}...")

        df = pd.read_csv(file_path)

        dataframes.append(df)

    master_df = pd.concat(dataframes, ignore_index=True)

    print("\nSuccessfully merged all CSV files.")

    return master_df

# ==========================================================
# Clean Metadata
# ==========================================================

def clean_metadata(df):
    """
    Remove unnecessary columns.
    """

    columns_to_drop = [
        "Unnamed: 0",
        "Codec",
        "File Size(MB)"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    print("Metadata cleaned successfully.")

    return df

# ==========================================================
# Save Metadata
# ==========================================================

def save_metadata(df):
    """
    Save cleaned metadata.
    """

    output_file = OUTPUT_FOLDER / "master_metadata.csv"

    df.to_csv(output_file, index=False)

    print(f"\nMetadata saved successfully!")
    print(f"Location: {output_file}")

# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "\nDataset not found!\n"
            "Download FaceForensics++ using the Lab setup notebook "
            "before running this script."
        )

    metadata = load_csv_files()

    cleaned_metadata = clean_metadata(metadata)

    save_metadata(cleaned_metadata)

    print("\nMetadata Engineering Completed Successfully!")

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()