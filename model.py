import torch.nn as nn
import torchvision
import torch

class Candens(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.densenet161(pretrained=True)
        self.backbone.classifier = nn.Identity()

        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

        self.logit = nn.Linear(2208, 1)
        
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.logit(x)   
        return x
        

