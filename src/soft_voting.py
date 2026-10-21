from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.dataset import CancerDataset
from src.transforms import eval_transform


# =========================================================
# 1. SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "patient_split_dataset"

TRAIN_DIR = DATASET_ROOT / "train cancer"
VALID_DIR = DATASET_ROOT / "valid cancer"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_CONFIGS = [
    ("convnext_tiny", "convnext_tiny_best.pth"),
    ("swin_t", "swin_t_best.pth"),
    ("densenet121", "densenet121_best.pth"),
    ("resnet50_improved", "resnet50_improved_best.pth"),
    ("efficientnet_b0", "efficientnet_b0_best.pth"),
]


# =========================================================
# 2. BUILD MODELS
# =========================================================

def build_model(model_name, num_classes):

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes,
        )

    elif model_name == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(
            model.head.in_features,
            num_classes,
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

    elif model_name == "resnet50_improved":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


def load_model(model_name, checkpoint_name, class_to_idx):

    checkpoint_path = MODELS_DIR / checkpoint_name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("class_to_idx") != class_to_idx:
        raise ValueError(
            f"Class mapping mismatch: {checkpoint_name}"
        )

    model = build_model(
        model_name,
        len(class_to_idx),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Loaded checkpoint:", checkpoint_name)

    return model


# =========================================================
# 3. PREDICT PROBABILITIES
# =========================================================

@torch.inference_mode()
def predict_probabilities(model, loader):

    all_probabilities = []
    all_labels = []

    for images, labels in loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# =========================================================
# 4. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("SOFT VOTING ENSEMBLE")
    print("=" * 60)
    print("Device:", DEVICE)

    # Class mapping comes from training data only.
    train_dataset = CancerDataset(
        TRAIN_DIR,
        transform=eval_transform,
    )

    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)

    if num_classes != 31:
        raise ValueError(
            f"Expected 31 classes, found {num_classes}"
        )

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    valid_dataset = CancerDataset(
        VALID_DIR,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )

    loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Match the stacking code's patient grouping.
    patient_ids = []

    for image_path, _ in valid_dataset.samples:

        relative_path = Path(
            image_path
        ).relative_to(VALID_DIR)

        if len(relative_path.parts) < 3:
            raise ValueError(
                "Expected patient_id/slide_id/image.jpg, "
                f"got: {relative_path}"
            )

        patient_ids.append(
            relative_path.parts[0]
        )

    groups = np.asarray(patient_ids)

    y = np.asarray(
        [label for _, label in valid_dataset.samples],
        dtype=np.int64,
    )

    # Reproduce the same deterministic fold selection
    # used by src.ensemble.
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_classes = set(np.unique(y))
    best_split = None
    best_coverage = -1

    for train_idx, valid_idx in splitter.split(
        np.zeros(len(y)),
        y,
        groups=groups,
    ):

        train_classes = set(np.unique(y[train_idx]))
        valid_classes = set(np.unique(y[valid_idx]))

        if train_classes != all_classes:
            continue

        coverage = len(valid_classes)

        if coverage > best_coverage:
            best_coverage = coverage
            best_split = (train_idx, valid_idx)

    if best_split is None:
        raise RuntimeError(
            "Could not reproduce the stacking validation split."
        )

    meta_train_idx, meta_valid_idx = best_split

    train_patients = set(groups[meta_train_idx])
    heldout_patients = set(groups[meta_valid_idx])

    if train_patients.intersection(heldout_patients):
        raise RuntimeError(
            "Patient overlap detected in held-out split."
        )

    print("Validation images:", len(y))
    print("Held-out images:", len(meta_valid_idx))
    print("Held-out patients:", len(heldout_patients))
    print("Patient overlap: 0")

    # Generate probabilities from all five models.
    probability_blocks = []

    for model_name, checkpoint_name in MODEL_CONFIGS:

        print(f"\nPredicting with {model_name}...")

        model = load_model(
            model_name,
            checkpoint_name,
            class_to_idx,
        )

        probabilities, labels = predict_probabilities(
            model,
            loader,
        )

        if not np.array_equal(labels, y):
            raise RuntimeError(
                f"Label order mismatch: {model_name}"
            )

        probability_blocks.append(probabilities)

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Average the five probability distributions.
    # Shape: (number of images, 31 classes)
    all_probabilities = np.mean(
        np.stack(probability_blocks, axis=0),
        axis=0,
    )

    # Evaluate only on the same held-out images as stacking.
    probabilities = all_probabilities[meta_valid_idx]
    y_true = y[meta_valid_idx]

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("SOFT VOTING RESULTS — HELD-OUT VALIDATION")
    print("=" * 60)

    print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:\n")

    print(
        classification_report(
            y_true,
            predictions,
            labels=list(range(num_classes)),
            target_names=[
                idx_to_class[i]
                for i in range(num_classes)
            ],
            zero_division=0,
        )
    )

    metrics = {
        "method": "soft_voting",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "validation_images": int(len(y)),
        "heldout_images": int(len(meta_valid_idx)),
        "heldout_patients": int(len(heldout_patients)),
        "patient_overlap": 0,
        "num_classes": num_classes,
        "models": [
            checkpoint_name
            for _, checkpoint_name in MODEL_CONFIGS
        ],
    }

    metrics_path = RESULTS_DIR / "soft_voting_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    np.savez_compressed(
        RESULTS_DIR / "soft_voting_predictions.npz",
        y_true=y_true,
        y_pred=predictions,
        probabilities=probabilities,
        patient_ids=groups[meta_valid_idx],
    )

    print("\nSaved metrics:", metrics_path)
    print(
        "Saved predictions:",
        RESULTS_DIR / "soft_voting_predictions.npz",
    )
    print("\nSoft voting completed successfully.")


if __name__ == "__main__":
    main()