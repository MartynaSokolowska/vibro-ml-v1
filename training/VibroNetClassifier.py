import torch.nn as nn
import torchvision.models as models


class VibroNetClassifier(nn.Module):
    """ResNet18-based audio classifier"""

    def __init__(self, num_classes=8):
        """
        Initialize ResNet18 classifier

        Args:
            num_classes: Number of temperature classes
        """
        super(VibroNetClassifier, self).__init__()

        self.resnet = models.resnet18(weights='DEFAULT')

        # Modify final layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)