import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch
import torch.nn.functional as F

import utils

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--exp', type=str, required=True, metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--data', type=str, default='bird_dataset_ex', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Define experiment folder
exp_dir = 'experiments/' + args.exp

# Load state
state = utils.load_state(exp_dir, get_best=True)

# Model
model = utils.get_model(state)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

from data import val_data_transforms

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file_path = exp_dir + '/kaggle.csv'
output_file = open(output_file_path, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        datas = val_data_transforms(pil_loader(test_dir + '/' + f))
        for i, data in enumerate(datas):
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                output = F.softmax(model(data), dim=1)
            res = output if i == 0 else res + output
        pred = res.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + output_file_path + ', you can upload this file to the kaggle competition website')
