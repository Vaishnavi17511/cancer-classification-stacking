
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

FILE_DIR = Path(__file__).resolve().parent

# This file is inside src/, so the project root is its parent.
PROJECT_ROOT = FILE_DIR.parent

MODEL_PATHS = {
    "convnext_tiny": PROJECT_ROOT / "models" / "convnext_tiny_best.pth",
    "swin_tiny": PROJECT_ROOT / "models" / "swin_t_best.pth",
    "densenet121": PROJECT_ROOT / "models" / "densenet121_best.pth",
    "resnet50_improved": PROJECT_ROOT / "models" / "resnet50_improved_best.pth",
    "efficientnet_b0": PROJECT_ROOT / "models" / "efficientnet_b0_best.pth",
}

RESULTS_ROOT = PROJECT_ROOT / "results" / "gradcam"

NUM_CLASSES = 31

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# 2. BUILD MODEL ARCHITECTURES
# Must match the architectures used during training.
# =========================================================

def build_model(model_name):

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            NUM_CLASSES,
        )

    elif model_name == "swin_tiny":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(
            model.head.in_features,
            NUM_CLASSES,
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES,
        )

    elif model_name == "resnet50_improved":
        # Assumes the trained model uses standard ResNet50
        # with a 31-class final fully connected layer.
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES,
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# =========================================================
# 3. LOAD THE PROJECT'S CLASS MAPPING
# =========================================================

_CLASS_TO_IDX = None


def load_class_mapping():

    global _CLASS_TO_IDX

    if _CLASS_TO_IDX is not None:
        return _CLASS_TO_IDX

    # Use a mapping saved in a checkpoint, if available.
    for checkpoint_path in MODEL_PATHS.values():

        if not checkpoint_path.is_file():
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            mapping = checkpoint.get("class_to_idx")

            if mapping is not None:
                mapping = {
                    str(label): int(index)
                    for label, index in mapping.items()
                }

                if len(mapping) != NUM_CLASSES:
                    raise ValueError(
                        "Checkpoint mapping does not contain "
                        f"{NUM_CLASSES} classes."
                    )

                _CLASS_TO_IDX = mapping
                return _CLASS_TO_IDX

    # Fallback: get the mapping from the training dataset.
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.dataset import create_dataloaders

    _, _, _, mapping = create_dataloaders(
        batch_size=1,
        num_workers=0,
    )

    _CLASS_TO_IDX = {
        str(label): int(index)
        for label, index in mapping.items()
    }

    if len(_CLASS_TO_IDX) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(_CLASS_TO_IDX)}."
        )

    return _CLASS_TO_IDX


# =========================================================
# 4. LOAD AND CACHE MODELS
# Each model is loaded only once per Python process.
# =========================================================

_LOADED_MODELS = {}


def load_model(model_name):

    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    checkpoint_path = MODEL_PATHS[model_name]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = build_model(model_name)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Get the common class mapping.
    current_mapping = load_class_mapping()

    # Verify the checkpoint mapping when it is available.
    if isinstance(checkpoint, dict):
        saved_mapping = checkpoint.get("class_to_idx")

        if saved_mapping is not None:

            saved_mapping = {
                str(label): int(index)
                for label, index in saved_mapping.items()
            }

            if saved_mapping != current_mapping:
                raise ValueError(
                    f"{model_name}: checkpoint class mapping "
                    "does not match the project's mapping."
                )

    idx_to_class = {
        index: label
        for label, index in current_mapping.items()
    }

    _LOADED_MODELS[model_name] = (model, idx_to_class)

    return model, idx_to_class


# =========================================================
# 5. SWIN TRANSFORMER RESHAPE
# =========================================================

def swin_reshape_transform(tensor):

    if tensor.ndim != 4:
        raise ValueError(
            "Unexpected Swin feature shape: "
            f"{tuple(tensor.shape)}"
        )

    # Convert NHWC to NCHW.
    return tensor.permute(0, 3, 1, 2)


# =========================================================
# 6. SELECT TARGET LAYER FOR GRAD-CAM
# =========================================================

def get_target_layers(model_name, model):

    if model_name == "convnext_tiny":
        return [model.features[-1]], None

    elif model_name == "swin_tiny":
        return [model.features[-1][-1].norm2], swin_reshape_transform

    elif model_name == "densenet121":
        return [model.features.denseblock4], None

    elif model_name == "resnet50_improved":
        return [model.layer4[-1]], None

    elif model_name == "efficientnet_b0":
        return [model.features[-1]], None

    raise ValueError(
        f"No Grad-CAM target layer configured for {model_name}"
    )


# =========================================================
# 7. PREPARE AN IMAGE
# Accepts a PIL image or a local file path.
# The frontend can pass its uploaded PIL image directly.
# =========================================================

def prepare_image(image):

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            image = opened_image.convert("RGB")

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL image or a local image path."
        )

    image = image.convert("RGB")

    resized_image = image.resize((224, 224))

    image_rgb = (
        np.asarray(resized_image, dtype=np.float32) / 255.0
    )

    input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    return image_rgb, input_tensor


# =========================================================
# 8. GENERATE GRAD-CAM FOR ONE MODEL
# =========================================================

def generate_gradcam(
    model_name,
    image,
    save_results=False,
    image_name="uploaded_image",
):
    """
    Args:
        model_name: A key from MODEL_PATHS.
        image: Uploaded PIL image or local image path.
        save_results: Save output images when True.
        image_name: Name used for saved output files.

    Returns:
        Prediction, confidence, probabilities, original image,
        heatmap and overlay as arrays.
    """

    model, idx_to_class = load_model(model_name)

    image_rgb, input_tensor = prepare_image(image)

    # 1. Predict the class.
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_idx = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[predicted_idx].item()
        )

    predicted_class = idx_to_class[predicted_idx]

    # 2. Generate Grad-CAM.
    target_layers, reshape_transform = get_target_layers(
        model_name,
        model,
    )

    cam_kwargs = {}

    if reshape_transform is not None:
        cam_kwargs["reshape_transform"] = reshape_transform

    # Grad-CAM requires gradients, so do not use no_grad or
    # inference_mode around this section.
    with GradCAM(
        model=model,
        target_layers=target_layers,
        **cam_kwargs,
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[
                ClassifierOutputTarget(predicted_idx)
            ],
        )[0]

    overlay = show_cam_on_image(
        image_rgb,
        grayscale_cam,
        use_rgb=True,
    )

    result = {
        "model": model_name,
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "confidence": confidence,
        "probabilities": probabilities.detach().cpu().numpy(),
        "class_to_idx": load_class_mapping(),
        "original": image_rgb,
        "heatmap": grayscale_cam,
        "overlay": overlay,
        "heatmap_path": None,
        "overlay_path": None,
        "comparison_path": None,
    }

    # 3. Optionally save images to disk.
    # Streamlit can display the returned arrays without saving.
    if save_results:

        import matplotlib.pyplot as plt

        output_dir = RESULTS_ROOT / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(image_name).stem or "uploaded_image"

        heatmap_path = output_dir / f"{safe_name}_heatmap.png"
        overlay_path = output_dir / f"{safe_name}_overlay.png"
        comparison_path = output_dir / f"{safe_name}_comparison.png"

        Image.fromarray(overlay).save(overlay_path)

        plt.imsave(
            heatmap_path,
            grayscale_cam,
            cmap="jet",
        )

        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 5),
        )

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")

        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")

        axes[2].imshow(overlay)
        axes[2].set_title("Heatmap Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{model_name}\n"
            f"Prediction: {predicted_class} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

        fig.tight_layout()

        fig.savefig(
            comparison_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        result["heatmap_path"] = heatmap_path
        result["overlay_path"] = overlay_path
        result["comparison_path"] = comparison_path

    return result


# =========================================================
# 9. GENERATE GRAD-CAM FOR ALL FIVE MODELS
# =========================================================

def generate_all_gradcams(
    image,
    save_results=False,
    image_name="uploaded_image",
):
    """
    Process the same uploaded image with all five models.

    If one model fails, the remaining models are still attempted.
    """

    results = {}
    errors = {}

    for model_name in MODEL_PATHS:

        try:
            results[model_name] = generate_gradcam(
                model_name=model_name,
                image=image,
                save_results=save_results,
                image_name=image_name,
            )

        except Exception as error:
            errors[model_name] = str(error)

    return {
        "results": results,
        "errors": errors,
        "device": str(DEVICE),
    }


# =========================================================
# 10. MODULE CHECK
# No image path is required.
# This does not load models or generate heatmaps.
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HISTOPATHOLOGY GRAD-CAM MODULE")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print("Available models:")

    for model_name, checkpoint_path in MODEL_PATHS.items():
        status = (
            "checkpoint found"
            if checkpoint_path.is_file()
            else "checkpoint missing"
        )

        print(f"  - {model_name}: {status}")

    print("\nModule check complete.")
    print("No image path is required.")
    print("Heatmaps will be generated when an image is supplied.")
