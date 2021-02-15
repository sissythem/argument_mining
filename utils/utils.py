import json
import random
import re
from datetime import datetime
from os.path import join

import numpy as np
import pandas as pd
from ellogon import tokeniser
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

    # ******************************* Oversampling **************************************************
    def oversample(self, task_kind, file_kind, total_num):
        """
        Oversampling of imbalanced datasets. For now it is implemented only for relations/stance datasets. The new
        datasets are exported into a csv file.

        Args
            task_kind (str): can take values from --> adu, rel or stance
            file_kind (str): possible values are --> train, test, dev
            total_num(int): the total number to augment the minority classes
        """
        filename = eval(f"self.{task_kind}_{file_kind}_csv")
        file_path = join(self.data_folder, filename)
        df = pd.read_csv(file_path, sep="\t", index_col=None, header=None)
        if task_kind == "adu":
            if not type(total_num) == dict:
                self.app_logger.warning(
                    "For ADU oversampling total_num should be a dict with the labels & their final count")
                self.app_logger.warning("Labels that have equal or greater number of instance will be kept the same")
            df = self.oversample_adus(data=df, desired_lbl_count=total_num)
        else:
            df = self.oversample_relations(df=df, rel=task_kind, total_num=total_num)
        filename = filename.replace(".csv", "")
        new_file = f"{filename}_oversample.csv"
        output_filepath = join(self.data_folder, new_file)
        df.to_csv(output_filepath, sep='\t', index=False, header=0)

    @staticmethod
    def oversample_adus(data, desired_lbl_count: dict):
        df_list = np.split(data, data[data.isnull().all(1)].index)
        df_list = [df.dropna() for df in df_list]
        labels_dict = {"B-major_claim": [], "I-major_claim": [], "B-claim": [], "I-claim": [], "B-premise": [],
                       "I-premise": []}
        for df in df_list:
            labels = list(df[1])
            for lbl in labels_dict.keys():
                if lbl in labels:
                    labels_dict[lbl].append(df)
        for lbl, desired_count in desired_lbl_count.items():
            dfs = labels_dict[lbl]
            append_idxs = []
            num_instances = len(dfs)
            ixs = range(num_instances)
            while True:
                if desired_count <= num_instances + len(append_idxs):
                    break
                num_append = min(desired_count - num_instances + len(append_idxs), num_instances)
                append_idxs.extend(random.sample(ixs, num_append))
            dfs.extend([dfs[i] for i in append_idxs])
        new_df_list = []
        for _, df_list in labels_dict.items():
            for df in df_list:
                empty_row = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]])
                new_df_list.append(df)
                new_df_list.append(empty_row)
        return pd.concat(new_df_list, axis=0)

    def oversample_relations(self, df, rel, total_num):
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
        labels = labels.squeeze()
        data = list(data)
        labels = list(labels)
        labels = [lbl_dict[lbl] for lbl in labels]
        data = [texts[x] for x in data]
        for idx, text in enumerate(data):
            text = text.replace("\t", " ")
            text = self.replace_multiple_spaces_with_single_space(text)
            data[idx] = text
        new_df = pd.DataFrame(columns=["text", "label"])
        new_df["text"] = data
        new_df["label"] = labels
        return new_df

    # *********************** Validation **************************************
    def save_invalid_json(self, document, validation_errors, invalid_adus):
        """
        Function to save an invalid json object into the output_files directory

        Args
            document (dict)
        """
        self.app_logger.debug("Writing invalid document into file")
        timestamp = datetime.now()
        filename = f"{document['id']}_{timestamp}.json"
        file_path = join(self.app_config.output_files, filename)
        with open(file_path, "w", encoding='utf8') as f:
            f.write(json.dumps(document, indent=4, sort_keys=False, ensure_ascii=False))
        with open(f"{file_path}.txt", "w") as f:
            for validation_error in validation_errors:
                f.write(validation_error.value + "\n")
            f.write(str(invalid_adus) + "\n")

    # **************************** Segment Extraction **************************************
    @staticmethod
    def get_label_with_max_conf(labels):
        max_lbl, max_conf = "", 0.0
        if labels:
            for label in labels:
                lbl = label.value
                conf = label.score
                if conf > max_conf:
                    max_lbl = lbl
                    max_conf = conf
        return max_lbl, max_conf

    @staticmethod
    def find_segment_in_text(content, text):
        try:
            start_idx = content.index(text)
        except(Exception, BaseException):
            try:
                start_idx = content.index(text[:10])
            except(Exception, BaseException):
                start_idx = None
        if start_idx:
            end_idx = start_idx + len(text)
        else:
            start_idx, end_idx = "", ""
        return start_idx, end_idx

    def get_args_from_sentence(self, sentence):
        if sentence.tokens:
            segments = []
            idx = None
            while True:
                segment, idx = self._get_next_segment(sentence.tokens, current_idx=idx)
                if segment:
                    segment["mean_conf"] = np.mean(segment["confidences"])
                    segments.append(segment)
                if idx is None:
                    break
            return segments

    def _get_next_segment(self, tokens, current_idx=None, current_label=None, segment=None):
        if current_idx is None:
            current_idx = 0
        if current_idx >= len(tokens):
            self.app_logger.debug("Sequence ended")
            return segment, None
        token = tokens[current_idx]
        raw_label = token.get_tag(label_type=None)
        lbl_txt = raw_label.value
        confidence = raw_label.score
        label_parts = lbl_txt.split("-")
        if len(label_parts) > 1:
            label_type, label = label_parts[0], label_parts[1]
        else:
            label_type, label = None, lbl_txt

        # if we're already tracking a contiguous segment:
        if current_label is not None:
            if label_type == "I" and current_label == label:
                # append to the running collections
                segment["text"] += " " + token.text
                segment["confidences"].append(confidence)
                return self._get_next_segment(tokens, current_idx + 1, current_label, segment)
            else:
                # new segment, different than the current one
                # next function call should start at the current_idx
                self.app_logger.debug(f"Returning completed segment: {segment['text']}")
                return segment, current_idx
        else:
            # only care about B-tags to start a segment
            if label_type == "B":
                segment = {"text": token.text, "label": label, "confidences": [confidence]}
                return self._get_next_segment(tokens, current_idx + 1, label, segment)
            else:
                return self._get_next_segment(tokens, current_idx + 1, None, segment)

    @staticmethod
    def get_adus(segments):
        major_claims = [(segment["segment"], segment["id"]) for segment in segments if segment["type"] == "major_claim"]
        claims = [(segment["segment"], segment["id"]) for segment in segments if segment["type"] == "claim"]
        premises = [(segment["segment"], segment["id"]) for segment in segments if segment["type"] == "premise"]
        return major_claims, claims, premises

    def concat_major_claim(self, segments, title, content, counter):
        if not segments:
            return []
        new_segments = []
        major_claim_txt = ""
        major_claims = [mc for mc in segments if mc["type"] == "major_claim"]
        mc_exists = False
        if major_claims:
            mc_exists = True
            for mc in major_claims:
                major_claim_txt += f" {mc['segment']}"
        else:
            major_claim_txt = title
        major_claim_txt = self.replace_multiple_spaces_with_single_space(text=major_claim_txt)
        already_found_mc = False
        if not mc_exists:
            counter += 1
            start_idx, end_idx = self.find_segment_in_text(content=content, text=major_claim_txt)
            major_claim = {
                "id": f"T{counter}",
                "type": "major_claim",
                "starts": str(start_idx),
                "ends": str(end_idx),
                "segment": major_claim_txt,
                "confidence": 0.99
            }
            new_segments.append(major_claim)
            already_found_mc = True
        for adu in segments:
            if adu["type"] == "major_claim":
                if not already_found_mc:
                    adu["segment"] = major_claim_txt
                    new_segments.append(adu)
                    already_found_mc = True
                else:
                    continue
            else:
                new_segments.append(adu)
        return new_segments

    # ******************************* Clustering ******************************************
    @staticmethod
    def collect_adu_for_clustering(documents, document_ids):
        # TODO uses only claims
        adus, adu_ids, doc_ids = [], [], []
        for document in documents:
            if document["id"] in document_ids:
                for adu in document["annotations"]["ADUs"]:
                    if adu["type"] == "claim":
                        adus.append(adu["segment"])
                        adu_ids.append(adu["id"])
                        doc_ids.append(document["id"])
        return adus, doc_ids, adu_ids

    # ******************************** Preprocessing *****************************************
    @staticmethod
    def is_old_annotation(attributes):
        for attribute in attributes:
            name = attribute["name"]
            if name == "premise_type" or name == "premise" or name == "claim":
                return True
        return False

    @staticmethod
    def collect_relation_pairs(parents, children, relation_pairs):
        new_relation_pairs = []
        count_relations = 0
        for p_id, p_text in parents.items():
            for c_id, c_text in children.items():
                key = (c_id, p_id)
                if key in relation_pairs.keys():
                    count_relations += 1
                relation = relation_pairs.get(key, "other")
                new_relation_pairs.append((c_text, p_text, relation))
        return new_relation_pairs

    # ***************************** Misc utilities functions ******************************
    @staticmethod
    def replace_multiple_spaces_with_single_space(text):
        return re.sub(' +', ' ', text)

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

    @staticmethod
    def get_greek_stopwords():
        return tokeniser.stop_words()

    @staticmethod
    def tokenize(text, punct=True):
        return tokeniser.tokenise_no_punc(text) if not punct else tokeniser.tokenise(text)
