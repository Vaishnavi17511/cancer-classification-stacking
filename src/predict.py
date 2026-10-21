
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from src.dataset import TRAIN_DIR
from src.transforms import eval_transform


# =========================================================
# 1. PROJECT SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_CONFIGS = {
    "ConvNeXt-Tiny": (
        "convnext_tiny",
        "convnext_tiny_best.pth",
    ),
    "Swin Transformer-Tiny": (
        "swin_t",
        "swin_t_best.pth",
    ),
    "DenseNet121": (
        "densenet121",
        "densenet121_best.pth",
    ),
    "ResNet50 Improved": (
        "resnet50_improved",
        "resnet50_improved_best.pth",
    ),
    "EfficientNet-B0": (
        "efficientnet_b0",
        "efficientnet_b0_best.pth",
    ),
}


# =========================================================
# 2. CLASS MAPPING
# =========================================================

def load_class_mapping():
    """
    Load the mapping used by the training dataset and
    verify it against each saved model checkpoint.
    """
    from src.dataset import CancerDataset

    dataset = CancerDataset(
        TRAIN_DIR,
        transform=eval_transform,
    )

    class_to_idx = dataset.class_to_idx

    if len(class_to_idx) != 31:
        raise ValueError(
            f"Expected 31 classes, got {len(class_to_idx)}"
        )

    return class_to_idx


# =========================================================
# 3. BUILD MODEL ARCHITECTURE
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


# =========================================================
# 4. LOAD A TRAINED MODEL
# =========================================================

def load_trained_model(
    model_name,
    checkpoint_name,
    class_to_idx,
):
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
            f"Class mapping mismatch in {checkpoint_name}"
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

    return model


# =========================================================
# 5. GET PROBABILITIES FOR ONE IMAGE
# =========================================================

@torch.inference_mode()
def model_probabilities(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    logits = model(image_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return probabilities[0].cpu().numpy()


# =========================================================
# 6. PREDICT USING ONE MODEL OR AN ENSEMBLE
# =========================================================

def predict_image(image, method="Soft Voting Ensemble"):
    """
    Parameters
    ----------
    image:
        PIL image supplied by the user.

    method:
        One of the five individual model names,
        'Soft Voting Ensemble', or 'Stacking Ensemble'.

    Returns
    -------
    Dictionary containing the predicted class,
    confidence, and all class probabilities.
    """

    class_to_idx = load_class_mapping()

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    # Apply the same deterministic preprocessing as evaluation.
    image = image.convert("RGB")
    image_tensor = eval_transform(image)

    # -----------------------------------------------------
    # A. INDIVIDUAL MODEL
    # -----------------------------------------------------

    if method in MODEL_CONFIGS:

        model_name, checkpoint_name = MODEL_CONFIGS[method]

        model = load_trained_model(
            model_name,
            checkpoint_name,
            class_to_idx,
        )

        probabilities = model_probabilities(
            model,
            image_tensor,
        )

        del model

    # -----------------------------------------------------
    # B. SOFT VOTING ENSEMBLE
    # -----------------------------------------------------

    elif method == "Soft Voting Ensemble":

        probability_blocks = []

        for model_name, checkpoint_name in MODEL_CONFIGS.values():

            model = load_trained_model(
                model_name,
                checkpoint_name,
                class_to_idx,
            )

            probabilities_one_model = model_probabilities(
                model,
                image_tensor,
            )

            probability_blocks.append(
                probabilities_one_model
            )

            del model

        probabilities = np.mean(
            np.stack(probability_blocks, axis=0),
            axis=0,
        )

    # -----------------------------------------------------
    # C. STACKING ENSEMBLE
    # -----------------------------------------------------

    elif method == "Stacking Ensemble":

        artifact_path = (
            RESULTS_DIR / "stacking_meta_learner.joblib"
        )

        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Stacking artifact not found: {artifact_path}"
            )

        artifact = joblib.load(artifact_path)

        if artifact["class_to_idx"] != class_to_idx:
            raise ValueError(
                "Stacking artifact class mapping mismatch."
            )

        if artifact["input_features"] != 155:
            raise ValueError(
                "Expected 155 stacking input features."
            )

        probability_blocks = []

        for model_name, checkpoint_name in MODEL_CONFIGS.values():

            model = load_trained_model(
                model_name,
                checkpoint_name,
                class_to_idx,
            )

            probabilities_one_model = model_probabilities(
                model,
                image_tensor,
            )

            probability_blocks.append(
                probabilities_one_model
            )

            del model

        # Preserve exactly the same feature order used
        # when training the meta-learner.
        features = np.concatenate(
            probability_blocks,
            axis=0,
        ).reshape(1, -1)

        meta_model = artifact["meta_model"]

        probabilities = meta_model.predict_proba(features)[0]

        # Align sklearn's probability columns to the
        # numerical class IDs used by the dataset.
        aligned_probabilities = np.zeros(
            len(class_to_idx),
            dtype=np.float64,
        )

        for column, class_id in enumerate(meta_model.classes_):
            aligned_probabilities[int(class_id)] = probabilities[column]

        probabilities = aligned_probabilities

    else:
        raise ValueError(
            f"Unsupported prediction method: {method}"
        )

    # -----------------------------------------------------
    # D. FORMAT THE RESULT
    # -----------------------------------------------------

    predicted_index = int(np.argmax(probabilities))
    predicted_class = idx_to_class[predicted_index]

    class_probabilities = {
        idx_to_class[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }

    top_indices = np.argsort(probabilities)[::-1][:5]

    top_predictions = [
        {
            "class": idx_to_class[int(index)],
            "probability": float(probabilities[index]),
        }
        for index in top_indices
    ]

    return {
        "model": method,
        "predicted_class": predicted_class,
        "confidence": float(probabilities[predicted_index]),
        "top_predictions": top_predictions,
        "class_probabilities": class_probabilities,
        "device": str(DEVICE),
    }
