import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
import random

class ResizeAug:
    def __init__(self, sizes, interpolation=Image.BILINEAR):
        self.sizes = sizes
        self.interpolation = interpolation

    def __call__(self, imgs):
        return [F.resize(img, (size, size), self.interpolation) for size in self.sizes for img in imgs]

class FlipAug:
    def __call__(self, imgs):
        return imgs + [F.hflip(img) for img in imgs]