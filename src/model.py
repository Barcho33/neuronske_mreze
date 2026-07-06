import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    model = get_resnet50(num_classes=8, pretrained=True, freeze_backbone=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("Output shape:", output.shape)