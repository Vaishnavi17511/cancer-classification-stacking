
import torch.nn as nn
from torchvision import models


def create_model(num_classes=31, pretrained=True):
    """
    Create a ResNet50 model for 31-class cancer classification.
    Uses ImageNet pretrained weights when requested.
    """

    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # Replace the original 1000-class output layer
    # with an output layer for our 31 cancer classes.
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model
