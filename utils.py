import os
from pathlib import Path

def get_raw_folder_names(raw_dir="data/raw"):
    """
    Extracts and returns a list of folder names from the raw data directory.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"Directory not found: {raw_dir}")
        return []
        
    # Get only directory names
    folder_names = [item.name for item in raw_path.iterdir() if item.is_dir()]
    return sorted(folder_names)

if __name__ == "__main__":
    folders = get_raw_folder_names()
    print(f"Found {len(folders)} folders in raw directory:")
    for folder in folders:
        print(f" - {folder}")
