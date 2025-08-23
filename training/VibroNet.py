import torch.nn as nn
import torchvision.models as models


class VibroNet(nn.Module):
    """ResNet18-based audio classifier / predictor"""

    def __init__(self, mode='classification', num_classes=8):
        """
        Args:
            mode (str): 'classification' or 'regression'
            num_classes: Number of temperature classes for classification
        """
        super(VibroNet, self).__init__()

        self.resnet = models.resnet18(weights='DEFAULT')

        num_features = self.resnet.fc.in_features
        if mode == 'regression':
            self.resnet.fc = nn.Linear(num_features, 1)
        else:
            self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)