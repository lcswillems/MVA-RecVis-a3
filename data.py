import zipfile
import os
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
import numpy
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    numpy.array,
    iaa.Sequential([
        # iaa.Sometimes(0.5,
        #     iaa.GaussianBlur(sigma=(0, 0.5))
        # ),
        # iaa.Sometimes(0.5,
        #     iaa.AdditiveGaussianNoise(scale=0.10*255),
        # ),
        iaa.ContrastNormalization((0.8, 1.2)),
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-25, 25),
            shear=(-20, 20)
        )
    ], random_order=True).augment_image,
    numpy.array,
    transforms.ToTensor(),
    normalize
])

val_data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])