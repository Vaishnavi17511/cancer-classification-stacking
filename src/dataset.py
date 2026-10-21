
import json
from pathlib import Path
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.transforms import train_transform, eval_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "patient_split_dataset"

TRAIN_DIR = DATASET_ROOT / "train cancer"
VALID_DIR = DATASET_ROOT / "valid cancer"
TEST_DIR = DATASET_ROOT / "test cancer"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
}


class CancerDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.root_dir}"
            )

        # Read labels from JSON and find matching images.
        for json_path in sorted(self.root_dir.rglob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as file:
                    label_data = json.load(file)

                label = label_data.get("label")

                if not isinstance(label, str) or not label.strip():
                    continue

                label = label.strip()
                image_path = None

                for candidate in json_path.parent.iterdir():
                    if (
                        candidate.is_file()
                        and candidate.stem == json_path.stem
                        and candidate.suffix.lower() in IMAGE_EXTENSIONS
                    ):
                        image_path = candidate
                        break

                if image_path is not None:
                    self.samples.append((image_path, label))

            except (OSError, json.JSONDecodeError) as error:
                print(f"Skipping {json_path}: {error}")

        if not self.samples:
            raise RuntimeError(
                f"No matching image/JSON pairs found in {self.root_dir}"
            )

        # Create class mapping from training data only.
        if class_to_idx is None:
            class_names = sorted(
                {label for _, label in self.samples}
            )
            self.class_to_idx = {
                name: index
                for index, name in enumerate(class_names)
            }
        else:
            self.class_to_idx = class_to_idx

        unknown_labels = sorted({
            label for _, label in self.samples
            if label not in self.class_to_idx
        })

        if unknown_labels:
            raise ValueError(
                "Labels missing from training class mapping: "
                f"{unknown_labels}"
            )

        self.classes = [
            name
            for name, index in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1]
            )
        ]

        # Store image path and numerical label.
        self.samples = [
            (image_path, self.class_to_idx[label])
            for image_path, label in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloaders(batch_size=32, num_workers=0):
    # Training dataset: random augmentation enabled.
    train_dataset = CancerDataset(
        TRAIN_DIR,
        transform=train_transform
    )

    class_to_idx = train_dataset.class_to_idx

    # Validation and test: no random augmentation.
    valid_dataset = CancerDataset(
        VALID_DIR,
        class_to_idx=class_to_idx,
        transform=eval_transform
    )

    test_dataset = CancerDataset(
        TEST_DIR,
        class_to_idx=class_to_idx,
        transform=eval_transform
    )

    # Count training samples per class.
    class_counts = Counter(
        label for _, label in train_dataset.samples
    )

    # Assign inverse-frequency weight to each training image.
    sample_weights = [
        1.0 / class_counts[label]
        for _, label in train_dataset.samples
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Weighted sampling is applied ONLY to training.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation/test retain their natural class distribution.
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        class_to_idx
    )


if __name__ == "__main__":
    train_dataset = CancerDataset(
        TRAIN_DIR,
        transform=train_transform
    )

    counts = Counter(
        label for _, label in train_dataset.samples
    )

    print("\n========== DATASET CHECK ==========")
    print("Training images:", f"{len(train_dataset):,}")
    print("Number of cancer classes:", len(train_dataset.classes))

    print("\nTraining images per class:")
    for class_name, class_index in sorted(
        train_dataset.class_to_idx.items(),
        key=lambda item: item[1]
    ):
        print(f"{class_name}: {counts[class_index]:,}")

    print("\n========== SAMPLER CHECK ==========")

    sample_weights = [
        1.0 / counts[label]
        for _, label in train_dataset.samples
    ]

    print("Number of sample weights:", len(sample_weights))
    print("Minimum sample weight:", min(sample_weights))
    print("Maximum sample weight:", max(sample_weights))

    print("\nDataset and weight calculation completed successfully.")
