from pathlib import Path

# Dataset location inside the repository
DATASET_PATH = Path("data/raw/ff-c23")

print("=" * 50)
print("FaceForensics++ Dataset Inspection")
print("=" * 50)

if not DATASET_PATH.exists():
    print(f"Dataset folder not found: {DATASET_PATH}")
    exit()

print("\nFolders found:\n")

for folder in sorted(DATASET_PATH.iterdir()):
    if folder.is_dir():
        print(f"- {folder.name}")