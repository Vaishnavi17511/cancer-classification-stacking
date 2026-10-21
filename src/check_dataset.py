
from pathlib import Path
import json
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent / "cancer dataset"

SPLITS = ["train cancer", "valid cancer", "test cancer"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

for split in SPLITS:
    folder = ROOT / split

    print(f"\n{'=' * 55}")
    print(f"SPLIT: {split}")
    print("Folder exists:", folder.is_dir())

    if not folder.is_dir():
        print("ERROR: Folder not found!")
        continue

    all_files = [p for p in folder.rglob("*") if p.is_file()]
    images = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    json_files = list(folder.rglob("*.json"))

    print("Total files:", len(all_files))
    print("Image files:", len(images))
    print("JSON files:", len(json_files))

    image_stems = {p.stem for p in images}
    json_stems = {p.stem for p in json_files}

    print("Images with matching JSON filename:",
          len(image_stems & json_stems))
    print("Images without matching JSON filename:",
          len(image_stems - json_stems))
    print("JSON without matching image filename:",
          len(json_stems - image_stems))

    labels = Counter()
    invalid_json = 0

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            label = data.get("label")
            if isinstance(label, str) and label.strip():
                labels[label.strip()] += 1
        except (OSError, json.JSONDecodeError):
            invalid_json += 1

    print("Valid JSON labels:", sum(labels.values()))
    print("Different labels:", len(labels))
    print("Invalid JSON files:", invalid_json)

print("\nDiagnostic completed.")
