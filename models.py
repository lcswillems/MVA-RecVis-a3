import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20

class TopModel(nn.Module):
    def __init__(self, base, last_layer_name):
        super().__init__()

        self.base = getattr(models, base)(pretrained=True)

        in_features = getattr(self.base, last_layer_name).in_features
        self.conv = nn.Conv2d(in_features, nclasses, 1)

        self.base = nn.Sequential(*list(self.base.children())[:-1])
        for weights in self.base.parameters():
            weights.requires_grad = False

    def forward(self, x):
        x = self.base(x)

        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])

        return x.squeeze(3).squeeze(2)

class Resnet(TopModel):
    def __init__(self, base):
        super().__init__(base, "fc")

class Resnet18(Resnet):
    def __init__(self):
        super().__init__("resnet18")

class Resnet34(Resnet):
    def __init__(self):
        super().__init__("resnet34")

class Resnet50(Resnet):
    def __init__(self):
        super().__init__("resnet50")

class Resnet101(Resnet):
    def __init__(self):
        super().__init__("resnet101")

class Resnet152(Resnet):
    def __init__(self):
        super().__init__("resnet152")

class Densenet(TopModel):
    def __init__(self, base):
        super().__init__(base, "classifier")

class Densenet161(Densenet):
    def __init__(self):
        super().__init__("densenet161")