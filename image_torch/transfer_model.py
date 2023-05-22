import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileCNN(nn.Module):
    def __init__(self, num_class=3) -> None:
        super(MobileCNN, self).__init__()
        
        self.backbone = mobilenet_v3_large(pretrained=True, progress=True)
        self.backbone.classifier[-1] = nn.Linear(1280, num_class)
        
    def forward(self, image):
        return self.backbone(image)