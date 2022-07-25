import torch
from torchvision import models as models


device = torch.device('cuda')
model = models.resnet50().to(device)

dummy_input = torch.randn(8,3,224,224, dtype=torch.float).to(device)

with torch.no_grad():
    _ = model(dummy_input)