import torch.nn as nn
import torchvision.models as models


class VibroNetRegressor(nn.Module):
    """ResNet18-based audio regressor"""

    def __init__(self):
        """
        Initialize ResNet18 regressor

        Args:
            num_classes: Number of temperature classes
        """
        super(VibroNetRegressor, self).__init__()

        self.resnet = models.resnet18(weights='DEFAULT')

        # Modify final layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)