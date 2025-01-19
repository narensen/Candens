import torch.nn as nn
import torchvision

class Candens(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.densenet161(pretrained=True)
        in_features = self.backbone.fc.in_features

        self.logit == nn.Linear(in_features)
