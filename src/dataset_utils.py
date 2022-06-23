import pandas as pd
import itertools
import os
import logging
from os.path import join
from datasets import Dataset
from datasets import load_dataset


def read_adu_csv(path):
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1].strip() for x in y] for y in split_list]
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})


def read_pairwise_csv(path):
    data = pd.read_csv(path, sep="\t", header=None)
    data.columns = ["text_pair", "label", "_"]
    if data["label"].isna().any():
        raise ValueError("Found NaN labels in the dataset!")
    return data


# def get_all_tokens_and_ner_tags(directory):
#     return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
#
# def get_un_token_dataset(train_directory, test_directory):
#     train_df = get_all_tokens_and_ner_tags(train_directory)
#     test_df = get_all_tokens_and_ner_tags(test_directory)
#     train_dataset = Dataset.from_pandas(train_df)
#     test_dataset = Dataset.from_pandas(test_df)
#
#     return (train_dataset, test_dataset)


def get_torch_dataset_from_path(path, fileloader_func):
    if os.path.isdir(path):
        logging.info(f"Reading all files from path {path}")
        # get all files within
        paths = [join(path, x) for x in os.listdir(path)]
    else:
        logging.info(f"Reading single file at path {path}")
        paths = [path]
    # read all with appropriate func and concat
    dfs = [fileloader_func(pth) for pth in paths]
    df = pd.concat(dfs).reset_index(drop=True)
    # convert to torch dataset
    return Dataset.from_pandas(df)


def get_pairwise_dataset(train_path, test_path):
    if os.path.isdir(train_path):
        train_dfs = [read_pairwise_csv(join(train_path, x)) for x in os.listdir(train_path)]
        test_dfs = [read_pairwise_csv(join(test_path, x)) for x in os.listdir(test_path)]
        train_dataset, test_dataset = Dataset.from_pandas(pd.concat(train_dfs)), Dataset.from_pandas(
            pd.concat(test_dfs))
    else:
        train_dataset = Dataset.from_pandas(read_pairwise_csv(train_path))
        test_dataset = Dataset.from_pandas(read_pairwise_csv(test_path))

    return train_dataset, test_dataset
