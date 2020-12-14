import hashlib
import json
import os
from os import environ, getcwd
from os.path import join
from pathlib import Path

import torch


def wrap_and_pad_tokens(inputs, prefix, suffix, seq_len, padding, pad=True, truncate=True):
    out = [prefix] + inputs
    if truncate:
        out = out[:seq_len - 1]
    out = out + [suffix]
    if pad:
        out += [padding] * (seq_len - 1 - len(out))
    return out


def add_padding(tokens, labels=None, max_len=512, pad_token=0):
    diff = max_len - tokens.shape[-1]
    if diff < 0:
        tokens = tokens[:, :max_len]
        if labels:
            labels = labels[:, :max_len]
    else:
        padding = torch.ones((1, diff), dtype=torch.long) * pad_token
        tokens = torch.cat([tokens, padding], dim=1)
        if labels:
            labels = torch.cat([labels, padding], dim=1)
    return tokens, labels


def remove_padding(tokens, predictions, pad_token=0):
    new_tokens, new_predictions = [], []
    for idx, token in enumerate(tokens):
        if (token != pad_token) and (token != 101) and (token != 102):
            new_tokens.append(token)
            new_predictions.append(predictions[idx])
    return torch.stack(new_tokens), torch.stack(new_predictions)


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


def get_base_path(path, base_name, hidden_size, layers, use_crf, optimizer, learning_rate, mini_batch_size):
    # Create a base path:
    embedding_names = 'bert-greek'
    base_path = "{}-".format(base_name) + '-'.join([
        str(embedding_names),
        'hs=' + str(hidden_size),
        'hl=' + str(layers),
        'crf=' + str(use_crf),
        "optmizer=" + optimizer,
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


def get_initial_json(name, text):
    hash_id = hashlib.md5(name.encode())
    return {
        "id": hash_id.hexdigest(),
        "link": "",
        "description": "",
        "date": "",
        "tags": [],
        "document_link": "",
        "publishedAt": "",
        "crawledAt": "",
        "domain": "",
        "netloc": "",
        "content": text,
        "annotations": {
            "ADUs": [],
            "Relations": []
        }
    }


def load_data(base_path, filename):
    filepath = join(base_path, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["data"]["documents"]
