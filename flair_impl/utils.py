import os
from os import environ, getcwd
from os.path import join
from pathlib import Path

import torch


def get_curr_path():
    curr_dir = Path(getcwd())
    parent = str(curr_dir.parent)
    curr_dir = str(curr_dir)
    return curr_dir if curr_dir.endswith("mining") else parent


def configure_device():
    if torch.cuda.is_available():
        devices = environ.get("CUDA_VISIBLE_DEVICES", 0)
        if type(devices) == str:
            devices = devices.split(",")
            device_name = "cuda:{}".format(devices[0].strip())
        else:
            device_name = "cuda:{}".format(devices)
    else:
        device_name = "cpu"
    return device_name


def get_base_path(path, hidden_size, rnn_layers, use_crf, optimizer, learning_rate, mini_batch_size):
    # Create a base path:
    embedding_names = 'bert-greek'
    base_path = 'model-' + '-'.join([
        str(embedding_names),
        'hs=' + str(hidden_size),
        'hl=' + str(rnn_layers),
        'crf=' + str(use_crf),
        str(optimizer.__name__),
        'lr=' + str(learning_rate),
        'bs=' + str(mini_batch_size)
    ])
    base_path = join(path, base_path)
    try:
        # os.mkdir(base_path, 0o755)
        os.makedirs(base_path)
    except (OSError, Exception):
        pass
    return base_path


def get_properties():
    # TODO load properties from file
    return {
        "do_prep": False,
        "hidden_size": 256,
        "rnn_layers": 2,
        "use_crf": True,
        "learning_rate": 0.0001,
        "mini_batch_size": 32,
        "num_workers": 8,
        "max_epochs": 150
    }
