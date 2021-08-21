from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.config import AppConfig
from collections import Counter


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val).squeeze() for key, val in self.encodings[idx].items()}
        # item['label'] = torch.tensor(self.labels[idx])
        item['labels'] = self.labels[idx]
        # return {"input_ids": self.encodings[idx], "labels": self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)


class ArgMiningDataset:

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.logger = app_config.app_logger

    @staticmethod
    def read_data(folder_path):
        data = {}
        # read data
        for x in "train dev test".split():
            path = join(folder_path, x + ".csv")
            data[x] = pd.read_csv(path, header=None, sep="\t")[[0, 1]]
        # join train & dev
        data["train"] = data["train"].append(data["dev"])
        del data["dev"]

        for k, v in data.items():
            # split
            tokens, labels = v[0].values.tolist(), v[1].values
            nans = [i for i, x in enumerate(tokens) if x is np.nan]
            # add first and last indexes
            nans = [-1] + nans + [-1]
            sentences = [tokens[nans[i]+1: nans[i + 1]] for i in range(len(nans) - 1)]
            labels = [labels[nans[i]+1: nans[i + 1]] for i in range(len(nans) - 1)]
            data[k] = (sentences, labels)
        if any(x is np.nan for l in labels for x in l):
            print("NaN found in labels")
            exit(1)
        if any(x is np.nan for l in sentences for x in l):
            print("NaN found in sentences")
            exit(1)
        return data["train"], data["test"]

    @staticmethod
    def pad_labels(labels, maxlen, dummy_label=-100):
        for i, label in enumerate(labels):
            label = label[:maxlen].tolist()
            if len(label) < maxlen:
                label += [dummy_label for _ in range(maxlen - len(label))]
            labels[i] = label
        if len(set(len(v) for v in labels)) != 1:
            raise ValueError("Label padding problematic -- different lengths encountered!")
        return labels

    @staticmethod
    def tokenize(data, tok, seqlen):
        return [tok(dat, is_split_into_words=True, return_tensors='pt', add_special_tokens=True, padding="max_length",
                    max_length=seqlen, truncation=True) for dat in data]

    def load_data(self, model_id, seqlen, limit_data=None):
        data_folder = join(self.app_config.dataset_folder, "adu")
        train, test = self.read_data(data_folder)
        if limit_data is not None:
            train = train[0][:limit_data], train[1][:limit_data]
            test = test[0][:limit_data], test[1][:limit_data]
        self.logger.info("Train label distribution")
        self.logger.info(Counter(np.concatenate(train[1])).most_common())
        self.logger.info("Train label distribution")
        self.logger.info(Counter(np.concatenate(test[1])).most_common())

        # tokenization
        self.logger.info(f"Tokenizing with {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id)
        self.logger.info(
            f"BOS {tok.bos_token_id}, EOS {tok.eos_token_id}, UNK {tok.unk_token_id} CLS {tok.cls_token_id} "
            f"MASK {tok.mask_token_id}")

        le = LabelEncoder()
        train_data = self.tokenize(train[0], tok, seqlen)
        train_labels = train[1]
        all_train_labels = np.concatenate(train_labels)
        le.fit_transform(all_train_labels)
        train_labels = [le.transform(tl) for tl in train_labels]
        token_labels = np.unique(np.concatenate(train_labels))

        self.logger.info("Tokenizing test")
        test_data = self.tokenize(test[0], tok, seqlen)
        test_labels = [le.transform(tl) for tl in test[1]]

        train_labels = self.pad_labels(train_labels, maxlen=seqlen)
        test_labels = self.pad_labels(test_labels, maxlen=seqlen)
        num_labels = len(token_labels)
        train_dset = MyDataset(train_data, train_labels)
        eval_dset = MyDataset(test_data, test_labels)
        return tok, num_labels, train_dset, eval_dset
