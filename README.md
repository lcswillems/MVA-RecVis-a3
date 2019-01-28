# Bird images classification

## Requirements

Run the following command to install required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is taken from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

The training/validation/test images can be downloaded [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip).

For some experiments, the birds need to be extracted from the image: execute the instructions in `colab_extract_birds.ipynb`.

## How to obtain the results?

These are the commands for training for each experiment:
- (2.1): `python train.py --exp dn161-1 --arch Densenet161 --data bird_dataset --no-train-data-aug --no-val-data-aug`
- (2.1)+(2.2): `python train.py --exp dn161-12 --arch Densenet161 --no-train-data-aug --no-val-data-aug`
- (2.1)+(2.2)+(2.3): `python train.py --exp dn161-123 --arch Densenet161 --no-val-data-aug`
- (2.1)+(2.2)+(2.3)+(2.4): `python train.py --exp dn161-1234 --arch Densenet161`

Then to evaluate:
```
python evaluate.py --exp dn161-1
```

You can also use the `colab_classify.ipynb` in the repo.

## Acknowledgments

Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.