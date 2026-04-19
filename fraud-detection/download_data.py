"""
download_data.py
================
Automatically downloads the Credit Card Fraud dataset from Kaggle
using the official kaggle Python package.

Kaggle credentials are read from (in order of priority):
  1. KAGGLE_API_TOKEN env var  (new Kaggle format — single token string)
  2. KAGGLE_USERNAME + KAGGLE_KEY env vars  (legacy format)
  3. ~/.config/kaggle/kaggle.json or ~/.kaggle/kaggle.json

One-time setup:
  - Go to kaggle.com -> Profile -> Settings -> API -> Generate New Token
  - Copy the token (starts with KGAT_...)
  - export KAGGLE_API_TOKEN=KGAT_...
  - Accept dataset rules at:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os
import json
import zipfile
from pathlib import Path

DATASET      = "mlg-ulb/creditcardfraud"
TARGET_FILE  = "data/creditcard.csv"
ZIP_PATH     = "data/creditcardfraud.zip"


def setup_credentials():
    """
    Write credentials to ~/.kaggle/kaggle.json so the kaggle package finds them.
    Supports both new single-token format and legacy username+key format.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Already configured — nothing to do
    if kaggle_json.exists():
        return

    # New format: single KAGGLE_API_TOKEN
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        print("  Configuring Kaggle credentials from KAGGLE_API_TOKEN...")
        kaggle_dir.mkdir(exist_ok=True)
        # The kaggle package accepts {"token": "KGAT_..."} in newer versions
        # but for maximum compatibility we also support username="__token__"
        kaggle_json.write_text(json.dumps({"token": token}))
        kaggle_json.chmod(0o600)
        return

    # Legacy format: separate env vars
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    if username and key:
        print("  Configuring Kaggle credentials from KAGGLE_USERNAME + KAGGLE_KEY...")
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        kaggle_json.chmod(0o600)
        return

    raise EnvironmentError(
        "\n"
        "  Kaggle credentials not found.\n\n"
        "  New token format (recommended):\n"
        "      export KAGGLE_API_TOKEN=KGAT_your_token_here\n\n"
        "  Legacy format:\n"
        "      export KAGGLE_USERNAME=your_username\n"
        "      export KAGGLE_KEY=your_api_key\n\n"
        "  Get your token: kaggle.com -> Profile -> Settings -> API\n"
        "  Then accept dataset rules at:\n"
        "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
    )


def download_dataset():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(TARGET_FILE):
        print(f"  Dataset already exists at {TARGET_FILE} -- skipping download.")
        return

    setup_credentials()

    print(f"  Downloading {DATASET} via Kaggle API...")
    print("  (~150 MB -- may take a minute)\n")

    try:
        import kaggle  # noqa — imported after credentials are set up
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET,
            path="data/",
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        err = str(e)
        if "403" in err or "forbidden" in err.lower():
            raise EnvironmentError(
                "  Access denied (403). You need to accept the dataset rules first:\n"
                "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            ) from e
        if "401" in err or "unauthorized" in err.lower():
            raise EnvironmentError(
                "  Invalid credentials (401). Check your KAGGLE_API_TOKEN."
            ) from e
        raise

    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(
            f"  Expected {TARGET_FILE} after extraction.\n"
            f"  Contents of data/: {os.listdir('data/')}"
        )

    size_mb = os.path.getsize(TARGET_FILE) / 1_048_576
    print(f"\n  Dataset ready: {TARGET_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_dataset()
