import torch.nn as nn
import torchvision.models as models


class VibroNet(nn.Module):
    """ResNet18-based audio classifier"""

    def __init__(self, num_classes=8):
        """
        Initialize ResNet18 classifier

        Args:
            num_classes: Number of temperature classes
            pretrained: Whether to use pretrained weights
        """
        super(VibroNet, self).__init__()

        # Load ResNet18
        self.resnet = models.resnet18(weights='DEFAULT')

        # Modify final layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Adaptive pooling to 224x224 for ResNet18
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))

    def forward(self, x):
        # Resize input to 224x224 for ResNet18
        x = self.adaptive_pool(x)
        return self.resnet(x)