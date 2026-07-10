from pathlib import Path

# ----------------------------
# Dataset Path
# ----------------------------

DATASET_PATH = Path("data/raw/FaceForensics++_C23")

# ----------------------------
# Check Dataset
# ----------------------------

if not DATASET_PATH.exists():
    print("Dataset not found!")
    exit()

print("=" * 60)
print("FaceForensics++ Dataset Inspection")
print("=" * 60)

print("\nDataset Location:")
print(DATASET_PATH)

print("\nFolders Found:\n")

folders = []

for folder in sorted(DATASET_PATH.iterdir()):

    if folder.is_dir():
      if folder.name != "csv":
  
        folders.append(folder.name)

        print(f"• {folder.name}")

print("\nTotal folders:", len(folders))