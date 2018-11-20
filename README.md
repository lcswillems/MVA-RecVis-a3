
## Object recognition and computer vision 2018/2019

### Assignment 3: Image classification

#### Requirements

1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### How to obtains the results?

To obtain:
- (1): `python train.py --exp dn161-1 --arch Densenet161 --data bird_dataset --no-train-data-aug --no-val-data-aug`
- (1)+(2): `python train.py --exp dn161-1 --arch Densenet161 --no-train-data-aug --no-val-data-aug`
- (1)+(2)+(3): `python train.py --exp dn161-1 --arch Densenet161 --no-val-data-aug`
- (1)+(2)+(3)+(4): `python train.py --exp dn161-1 --arch Densenet161`

#### Dataset

We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model

Run the script `train.py` to train your model.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Acknowledgments

Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.