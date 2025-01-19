import torch.nn as nn
import torchvision
import torch

class Candens(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.densenet161(pretrained=True)
        in_features = self.backbone.classifier.in_features

        self.logit == nn.Linear(in_features, 1)
        
    def forward(self, x):
        x = self.backbone.features(x)
        x = torch.flatten(x)

        x = self.logit(x)
        return x
        

