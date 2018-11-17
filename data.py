import zipfile
import os

import torchvision.transforms as transforms

size = 224

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_data_transforms = transforms.Compose([
    transforms.Resize(int(size*1.14)),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    normalize
])