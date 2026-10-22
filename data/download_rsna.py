"""
RSNA Lumbar Spine Degeneration Classification Dataset Downloader.
Supports Kagglehub and Kaggle API downloading for the official RSNA competition data.
"""
import os
import shutil
from pathlib import Path
from config.config import CONFIG

COMPETITION_NAME = "rsna-2024-lumbar-spine-degeneration-classification"

def download_rsna_from_kaggle(destination_dir: Path = None):
    """
    Downloads the official RSNA dataset using kagglehub or the kaggle API.
    Requirements:
      - Kaggle API key (KAGGLE_USERNAME and KAGGLE_KEY environment variables or ~/.kaggle/kaggle.json)
    """
    if destination_dir is None:
        destination_dir = CONFIG.data_dir

    print(f"Checking for Kaggle credentials to download {COMPETITION_NAME}...")
    
    try:
        import kagglehub
        print(f"Downloading {COMPETITION_NAME} via kagglehub...")
        download_path = kagglehub.competition_download(COMPETITION_NAME)
        print(f"Downloaded files to cache at: {download_path}")
        
        # Link or copy to destination_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        print(f"Symlinking/copying dataset files to {destination_dir}...")
        for item in Path(download_path).iterdir():
            target = destination_dir / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        print("RSNA dataset successfully placed in target directory.")
        return True
    except Exception as e:
        print(f"\n[Note] Kaggle download encountered: {e}")
        print("\nTo use the official RSNA dataset:")
        print("1. Set your KAGGLE_USERNAME and KAGGLE_KEY environment variables, or place kaggle.json in ~/.kaggle/")
        print("2. Run: kaggle competitions download -c rsna-2024-lumbar-spine-degeneration-classification")
        print("3. Unzip the downloaded archive into: data/dataset/")
        print("\nFor rapid prototyping and testing, use the included synthetic data generator: python -m data.synthetic_data_generator\n")
        return False

if __name__ == "__main__":
    download_rsna_from_kaggle()
