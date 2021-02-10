from os.path import join

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from utils.config import AppConfig


class Utilities:
    """
    Various utility methods
    """

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.data_folder = join(app_config.resources_path, "data")

        self.adu_train_csv = app_config.adu_train_csv
        self.adu_dev_csv = app_config.adu_dev_csv
        self.adu_test_csv = app_config.adu_test_csv

        self.rel_train_csv = app_config.rel_train_csv
        self.rel_dev_csv = app_config.rel_dev_csv
        self.rel_test_csv = app_config.rel_test_csv

        self.stance_train_csv = app_config.stance_train_csv
        self.stance_dev_csv = app_config.stance_dev_csv
        self.stance_test_csv = app_config.stance_test_csv

    def oversample(self, task_kind, file_kind, total_num):
        filename = eval(f"self.{task_kind}_{file_kind}_csv")
        file_path = join(self.data_folder, filename)
        df = pd.read_csv(file_path, sep="\t", index_col=None, header=None)
        if task_kind == "adu":
            pass
        else:
            df = self._relation_oversampling(df=df, rel=task_kind, total_num=total_num)
        filename = filename.replace(".csv", "")
        new_file = f"{filename}_oversample.csv"
        output_filepath = join(self.data_folder, new_file)
        df.to_csv(output_filepath, sep='\t', index=False, header=0)

    @staticmethod
    def _relation_oversampling(df, rel, total_num):
        texts = list(df[0])
        indices = [texts.index(x) for x in texts]
        labels = list(df[1])
        numeric_labels = []
        unique_labels = set(labels)
        count = 0
        lbl_dict = {}
        str_to_num = {}
        for lbl in unique_labels:
            lbl_dict[count] = lbl
            str_to_num[lbl] = count
            count += 1
        for lbl in labels:
            numeric_labels.append(str_to_num[lbl])
        data = np.asarray(indices).reshape(-1, 1)
        labels = np.asarray(numeric_labels).reshape(-1, 1)
        if rel == "rel":
            num_support = str_to_num["support"]
            num_attack = str_to_num["attack"]
            sampling_strategy = {num_support: total_num, num_attack: total_num}
        else:
            num_against = str_to_num["against"]
            sampling_strategy = {num_against: total_num}
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy)
        data, labels = sampler.fit_resample(data.reshape(-1, 1), labels)
        data = data.squeeze()
        new_df = pd.DataFrame(columns=["text", "label"])
        new_df["text"] = list(data)
        new_df["label"] = list(labels)
        return new_df

    def name_exceeds_bytes(self, name):
        """
        Checks if a string exceeds the 255 bytes

        Args
            name (str): the name of a file

        Returns
            bool: True/False
        """
        return self._utf8len(name) >= 255

    @staticmethod
    def _utf8len(s):
        """
        Find the length of the encoded filename

        Args
            s (str): the filename to encode

        Returns
            int: the length of the encoded filename
        """
        return len(s.encode('utf-8'))
