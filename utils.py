import os
import logging
import sys
import torch
import shutil

import models

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def get_log_path(exp_dir):
    return exp_dir + '/log.txt'

def get_logger(exp_dir):
    path = get_log_path(exp_dir)
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_chkpt_path(exp_dir):
    return exp_dir + '/last.pth.tar'

def get_best_chkpt_path(exp_dir):
    return exp_dir + '/best.pth.tar'

def load_state(exp_dir, get_best=False):
    filepath = get_best_chkpt_path(exp_dir) if get_best else get_chkpt_path(exp_dir)

    return torch.load(filepath, map_location='cpu')

def get_model(state):
    model = getattr(models, state['arch'])()
    if 'model_state' in state.keys():
        model.load_state_dict(state['model_state'])
    return model

def save_state(state, is_best, exp_dir):
    filepath = get_chkpt_path(exp_dir)
    create_folders_if_necessary(filepath)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, get_best_chkpt_path(exp_dir))