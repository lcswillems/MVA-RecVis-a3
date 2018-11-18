import zipfile
import os

import torchvision.transforms as transforms

size = 256
crop_size = 224

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(crop_size, scale=(.6, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.4, .4, .4),
    transforms.ToTensor(),
    normalize
])

val_data_transforms = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    normalize
])