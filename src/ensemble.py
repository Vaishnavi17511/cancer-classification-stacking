from pathlib import Path
import json
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.dataset import CancerDataset
from src.transforms import eval_transform


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
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
# 2. BUILD MODEL ARCHITECTURES
# =========================================================

def build_model(model_name, num_classes):
    """Build the architecture matching the saved checkpoint."""

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(
            features, num_classes
        )

    elif model_name == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(
            model.head.in_features, num_classes
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        features = model.classifier.in_features
        model.classifier = nn.Linear(
            features, num_classes
        )

    elif model_name == "resnet50_improved":
        model = models.resnet50(weights=None)
        features = model.fc.in_features
        model.fc = nn.Linear(
            features, num_classes
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            features, num_classes
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


def load_checkpoint(model_name, checkpoint_name, class_to_idx):
    """Load a trained model and verify its class mapping."""

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

    saved_mapping = checkpoint.get("class_to_idx")

    if saved_mapping != class_to_idx:
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

    print(f"Loaded checkpoint: {checkpoint_name}")

    return model


# =========================================================
# 3. GENERATE MODEL PROBABILITIES
# =========================================================

@torch.inference_mode()
def predict_probabilities(model, loader):
    """Generate class probabilities for all images."""

    probability_batches = []
    label_batches = []

    for images, labels in loader:
        images = images.to(DEVICE)

        logits = model(images)
        probabilities = torch.softmax(
            logits, dim=1
        )

        probability_batches.append(
            probabilities.cpu().numpy()
        )

        label_batches.append(
            labels.numpy()
        )

    return (
        np.concatenate(probability_batches, axis=0),
        np.concatenate(label_batches, axis=0),
    )


# =========================================================
# 4. MAIN STACKING ENSEMBLE
# =========================================================

def main():

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("=" * 60)
    print("STACKING ENSEMBLE")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Dataset:", DATASET_ROOT)

    # Check required folders.
    for folder in (TRAIN_DIR, VALID_DIR):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Required dataset folder not found: {folder}"
            )

    # -----------------------------------------------------
    # Load training dataset to obtain class mapping.
    # -----------------------------------------------------

    train_dataset = CancerDataset(
        TRAIN_DIR,
        transform=eval_transform,
    )

    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)

    if num_classes != 31:
        raise ValueError(
            f"Expected 31 classes; found {num_classes}"
        )

    print("Number of classes:", num_classes)

    # -----------------------------------------------------
    # Load the existing cleaned validation dataset.
    # No dataset cleaning or splitting is performed here.
    # -----------------------------------------------------

    valid_dataset = CancerDataset(
        VALID_DIR,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # -----------------------------------------------------
    # Extract patient IDs.
    #
    # Expected structure:
    # valid cancer / PATIENT_ID / SLIDE_ID / image.jpg
    # -----------------------------------------------------

    patient_ids = []

    for image_path, _ in valid_dataset.samples:

        relative_path = Path(
            image_path
        ).relative_to(VALID_DIR)

        if len(relative_path.parts) < 3:
            raise ValueError(
                "Unexpected dataset structure. Expected "
                "patient_id/slide_id/image.jpg, got: "
                f"{relative_path}"
            )

        patient_ids.append(
            relative_path.parts[0]
        )

    patient_ids = np.asarray(patient_ids)

    y = np.asarray(
        [label for _, label in valid_dataset.samples],
        dtype=np.int64,
    )

    if len(np.unique(patient_ids)) < 2:
        raise ValueError(
            "Validation data must contain at least two patients."
        )

    # -----------------------------------------------------
    # Generate predictions using all five trained models.
    # -----------------------------------------------------

    probability_blocks = []

    for model_name, checkpoint_name in MODEL_CONFIGS:

        print(
            f"\nGenerating predictions: {model_name}"
        )

        model = load_checkpoint(
            model_name,
            checkpoint_name,
            class_to_idx,
        )

        probabilities, predicted_labels = (
            predict_probabilities(
                model,
                valid_loader,
            )
        )

        if not np.array_equal(predicted_labels, y):
            raise RuntimeError(
                f"Label ordering mismatch for {model_name}"
            )

        expected_shape = (
            len(y),
            num_classes,
        )

        if probabilities.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected probability shape for "
                f"{model_name}: {probabilities.shape}"
            )

        probability_blocks.append(
            probabilities
        )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Five models × 31 classes = 155 features per image.
    X = np.concatenate(
        probability_blocks,
        axis=1,
    )

    groups = patient_ids

    print("\nCombined feature shape:", X.shape)
    print("Validation images:", len(y))
    print(
        "Validation patients:",
        len(np.unique(groups)),
    )

    # -----------------------------------------------------
    # PATIENT-AWARE META-LEARNER SPLIT
    #
    # StratifiedGroupKFold attempts to preserve class
    # distribution while keeping patients in one split.
    # The cleaned dataset itself is not changed.
    # -----------------------------------------------------

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_classes = set(np.unique(y))

    best_split = None
    best_coverage = -1

    for train_idx, valid_idx in splitter.split(
        X,
        y,
        groups=groups,
    ):

        train_classes = set(
            np.unique(y[train_idx])
        )

        valid_classes = set(
            np.unique(y[valid_idx])
        )

        # Meta-training must include every class.
        if train_classes != all_classes:
            continue

        # Select the fold with the greatest held-out
        # class coverage among eligible folds.
        coverage = len(valid_classes)

        if coverage > best_coverage:
            best_coverage = coverage

            best_split = (
                train_idx,
                valid_idx,
            )

    if best_split is None:
        raise RuntimeError(
            "Could not find a patient-grouped split with "
            "all classes in meta-training. Try a different "
            "split configuration."
        )

    meta_train_idx, meta_valid_idx = best_split

    train_patients = set(
        groups[meta_train_idx]
    )

    heldout_patients = set(
        groups[meta_valid_idx]
    )

    overlap = train_patients.intersection(
        heldout_patients
    )

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: "
            f"{len(overlap)} overlapping patients."
        )

    train_classes = set(
        np.unique(y[meta_train_idx])
    )

    valid_classes = set(
        np.unique(y[meta_valid_idx])
    )

    missing_valid_classes = (
        all_classes - valid_classes
    )

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    missing_valid_names = [
        idx_to_class[index]
        for index in sorted(missing_valid_classes)
    ]

    print("\nMeta-training images:", len(meta_train_idx))
    print("Held-out images:", len(meta_valid_idx))

    print(
        "Meta-training patients:",
        len(train_patients),
    )

    print(
        "Held-out patients:",
        len(heldout_patients),
    )

    print(
        "Meta-training classes:",
        len(train_classes),
    )

    print(
        "Held-out classes:",
        len(valid_classes),
    )

    print("Patient overlap:", len(overlap))

    if missing_valid_names:
        print(
            "Classes absent from held-out subset:",
            missing_valid_names,
        )
    else:
        print(
            "All 31 classes are represented "
            "in the held-out subset."
        )

    # -----------------------------------------------------
    # TRAIN LOGISTIC REGRESSION META-LEARNER
    # -----------------------------------------------------

    meta_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    print(
        "\nTraining Logistic Regression meta-learner..."
    )

    meta_model.fit(
        X[meta_train_idx],
        y[meta_train_idx],
    )

    # Evaluate on held-out validation patients.
    predictions = meta_model.predict(
        X[meta_valid_idx]
    )

    y_true = y[meta_valid_idx]

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
    print("STACKING ENSEMBLE RESULTS")
    print("=" * 60)

    print(
        f"Accuracy:    {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

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

    # -----------------------------------------------------
    # SAVE META-LEARNER AND RESULTS
    # -----------------------------------------------------

    artifact = {
        "meta_model": meta_model,
        "class_to_idx": class_to_idx,
        "base_models": MODEL_CONFIGS,
        "num_classes": num_classes,
        "input_features": int(X.shape[1]),
        "image_size": 224,
        "normalization_mean": [
            0.485, 0.456, 0.406
        ],
        "normalization_std": [
            0.229, 0.224, 0.225
        ],
    }

    artifact_path = (
        RESULTS_DIR / "stacking_meta_learner.joblib"
    )

    joblib.dump(
        artifact,
        artifact_path,
    )

    np.savez_compressed(
        RESULTS_DIR / "stacking_validation_predictions.npz",
        y_true=y_true,
        y_pred=predictions,
        patient_ids=groups[meta_valid_idx],
        probabilities=X[meta_valid_idx],
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "meta_train_images": int(len(meta_train_idx)),
        "heldout_images": int(len(meta_valid_idx)),
        "meta_train_patients": int(len(train_patients)),
        "heldout_patients": int(len(heldout_patients)),
        "patient_overlap": int(len(overlap)),
        "meta_train_classes": int(len(train_classes)),
        "heldout_classes": int(len(valid_classes)),
        "classes_absent_from_heldout": missing_valid_names,
    }

    metrics_path = (
        RESULTS_DIR / "stacking_validation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nSaved meta-learner:", artifact_path)
    print("Saved validation predictions.")
    print("Saved validation metrics:", metrics_path)
    print("\nStacking workflow completed.")


if __name__ == "__main__":
    main()