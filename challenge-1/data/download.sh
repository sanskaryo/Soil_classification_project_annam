#!/bin/bash

# === Example: Download competition data ===
KAGGLE_COMPETITION="soil-classification"
TARGET_DIR="./data"

echo "Downloading competition data: $KAGGLE_COMPETITION"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
