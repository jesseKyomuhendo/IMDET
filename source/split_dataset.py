"""
split_dataset.py
----------------
Splits CASIA 2.0 into train / val / test sets and saves the file paths
as CSV files in the splits directory defined in config.yaml.

Does NOT copy or move any images. The CSV files contain:
    filepath  : relative path to the image
    label     : class name (authentic, copy_move, splicing)

Output:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Usage:
    Press IDE run button or run the following CLI command: python source/split_dataset.py
"""

import csv
import random
from pathlib import Path
import os
from settings.SettingsAssistant import CONFIG


def get_images(class_dir: Path, label: str):
    """Return list of (filepath, label) for all images in a class folder."""
    extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".png"}
    return [
        (str(f), label)
        for f in sorted(class_dir.iterdir())
        if f.is_file() and f.suffix.lower() in extensions
    ]


def split(items, train_ratio, val_ratio, seed):
    """Shuffle and split a list into train / val / test."""
    random.seed(seed)
    random.shuffle(items)
    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)
    print(f"  Saved {len(rows):>5} rows → {path}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    casia_dir  = Path(CONFIG["data"]["casia_dir"])
    split_dir  = Path(CONFIG["data"]["split_dir"])
    classes    = CONFIG["classes"]
    train_r    = CONFIG["split"]["train"]
    val_r      = CONFIG["split"]["val"]
    seed       = CONFIG["split"]["seed"]

    print(f"\nCASIA directory : {casia_dir.resolve()}")
    print(f"Classes         : {classes}")
    print(f"Split ratios    : train={train_r}  val={val_r}  test={round(1-train_r-val_r, 2)}")
    print(f"Random seed     : {seed}\n")

    all_train, all_val, all_test = [], [], []

    for label in classes:
        class_dir = casia_dir / label
        if not class_dir.exists():
            print(f"Warning: Class folder not found: {class_dir}")
            continue

        images = get_images(class_dir, label)
        train, val, test = split(images, train_r, val_r, seed)

        print(f"  {label:<12} total={len(images):>5}  "
              f"train={len(train):>4}  val={len(val):>4}  test={len(test):>4}")

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    # Shuffle combined splits so batches have mixed classes
    random.seed(seed)
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)

    print()
    save_csv(all_train, split_dir / "train.csv")
    save_csv(all_val,   split_dir / "val.csv")
    save_csv(all_test,  split_dir / "test.csv")

    print(f"\nDone. Total images: "
          f"train={len(all_train)}  val={len(all_val)}  test={len(all_test)}")


if __name__ == "__main__":
    main()