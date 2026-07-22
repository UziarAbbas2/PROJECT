"""
Download the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle.

Prerequisites:
  1. pip install kaggle
  2. Create a Kaggle API token: https://www.kaggle.com/settings -> "Create New Token"
     This downloads kaggle.json. Place it at ~/.kaggle/kaggle.json
     (on Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json) and run:
         chmod 600 ~/.kaggle/kaggle.json

Usage:
    python download_dataset.py
"""

import os
import subprocess
import zipfile

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
DEST_DIR = "."
ZIP_NAME = "chest-xray-pneumonia.zip"


def main():
    print(f"Downloading dataset: {DATASET_SLUG}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", DEST_DIR],
        check=True,
    )

    zip_path = os.path.join(DEST_DIR, ZIP_NAME)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST_DIR)

    print("Done. Expected folder structure:")
    print("  ./chest_xray/train/{NORMAL,PNEUMONIA}")
    print("  ./chest_xray/val/{NORMAL,PNEUMONIA}")
    print("  ./chest_xray/test/{NORMAL,PNEUMONIA}")


if __name__ == "__main__":
    main()
