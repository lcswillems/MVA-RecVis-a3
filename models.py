import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Resnet(nn.Module):
    def __init__(self, base):
        super().__init__()

        self.base = getattr(models, base)(pretrained=True)

        in_features = self.base.fc.in_features

        self.base = nn.Sequential(*list(self.base.children())[:-1])

        for weights in self.base.parameters():
            weights.requires_grad = False

        self.fc = nn.Linear(in_features, nclasses)
        self.conv = nn.Conv2d(in_features, nclasses, 1)

    def eval(self):
        super().eval()

        self.update_conv_weights()

    def update_conv_weights(self):
        self.conv.weight = nn.Parameter(self.fc.weight.view(*self.fc.weight.size(), 1, 1))

    def forward(self, x):
        x = self.base(x)

        if self.training:
            x = x.view(x.size(0), -1)
            return self.fc(x)

        x = self.conv(x)
        x = torch.max(x, (2, 3))
        return x

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