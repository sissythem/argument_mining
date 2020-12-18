import json
from os.path import join
from typing import List

from elasticsearch_dsl import Search
from ellogon import tokeniser
from flair.data import Sentence, Label
# import yaml

import utils
from classifiers import AduModel, RelationsModel
from utils import AppConfig


class ArgumentMining:

    def __init__(self, app_config):
        self.app_config: AppConfig = app_config
        self.app_logger = app_config.app_logger

        # load ADU model
        self.adu_model = AduModel(app_config=self.app_config)
        self.adu_model.load()

        # load Relations model
        self.rel_model = RelationsModel(app_config=self.app_config, dev_csv=self.app_config.rel_dev_csv,
                                        train_csv=self.app_config.rel_train_csv, test_csv=self.app_config.rel_test_csv,
                                        base_path=self.app_config.rel_base_path, model_name="rel")
        self.rel_model.load()

        # load Stance model
        self.stance_model = RelationsModel(app_config=self.app_config, dev_csv=self.app_config.stance_dev_csv,
                                           train_csv=self.app_config.stance_train_csv,
                                           test_csv=self.app_config.stance_test_csv,
                                           base_path=self.app_config.stance_base_path, model_name="stance")
        self.stance_model.load()

    def run_pipeline(self):
        eval_source = self.app_config.properties["eval"]["source"]
        eval_target = self.app_config.properties["eval"]["target"]
        if eval_source == "elasticsearch":
            documents, ids = self._retrieve_from_elasticsearch()
        else:
            documents, ids = self._retrieve_from_file()
        for document in documents:
            document = self.predict(document=document)
            if eval_target == "elasticsearch":
                self.app_config.elastic_save.elasticsearch_client.create(index='debatelab', ignore=400,
                                                                         doc_type='docket', id=document["id"],
                                                                         body=document)
            else:
                filename = document["title"] + ".json"
                if utils.name_exceeds_bytes(filename):
                    filename = document["id"] + ".json"
                file_path = join(self.app_config.out_files_path, filename)
                with open(file_path, "w", encoding='utf8') as f:
                    f.write(json.dumps(document, indent=4, sort_keys=False, ensure_ascii=False))
        self.app_logger.info("Evaluation is finished!")

    def _retrieve_from_elasticsearch(self):
        documents, ids = [], []
        client = self.app_config.elastic_retrieve.elasticsearch_client

        file_path = join(self.app_config.resources_path, "kasteli_34_urls.txt")
        # read the list of urls from the file:
        with open(file_path, "r") as f:
            urls = [line.rstrip() for line in f]
        search_articles = Search(using=client, index='articles').filter('terms', link=urls)

        # TODO retrieve previous day's articles, save now to properties
        # now = datetime.now()
        # previous_date = app_config.properties["eval"]["last_date"]
        # result = Search(using=client, index='articles').filter('range', date={'gt': previous_date, 'lte': now})
        for hit in search_articles.scan():
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            if not document["content"].startswith(document["title"]):
                document["content"] = document["title"] + "\r\n\r\n" + document["content"]
            ids.append(document["id"])
            documents.append(document)

        # update last search date
        # properties = self.app_config.properties
        # properties["eval"]["last_date"] = now
        # with open(join(self.app_config.resources_path, self.app_config.properties_file), "w") as f:
            # yaml.dump(properties, f)
        return documents, ids

    def _retrieve_from_file(self, filename="kasteli.json"):
        documents, ids = [], []
        self.app_logger.info("Evaluating using file: {}".format(filename))
        file_path = join(self.app_config.resources_path, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
        data = data["data"]["documents"]
        if data:
            for document in data:
                document = utils.get_initial_json(document["name"], document["text"])
                documents.append(document)
                ids.append(document["id"])
        return documents, ids

    def predict(self, document):
        json_obj = self._predict_adus(document=document)
        segments = json_obj["annotations"]["ADUs"]
        major_claims, claims, premises = self._get_adus(segments)
        json_obj = self._predict_relations(major_claims=major_claims, claims=claims, premises=premises,
                                           json_obj=json_obj)
        json_obj = self._predict_stance(major_claims=major_claims, claims=claims, json_obj=json_obj)
        return json_obj

    def _predict_adus(self, document):
        # init document id & annotations
        document["annotations"] = {
            "ADUs": [],
            "Relations": []
        }

        self.app_logger.debug(
            "Processing document with id: {} and name: {}".format(document["id"], document["title"]))

        segment_counter = 0
        sentences = tokeniser.tokenise_no_punc(document["content"])
        for sentence in sentences:
            self.app_logger.debug("Predicting labels for sentence: {}".format(sentence))
            sentence = " ".join(list(sentence))
            sentence = Sentence(sentence)
            self.adu_model.model.predict(sentence, all_tag_prob=True)
            self.app_logger.debug("Output: {}".format(sentence.to_tagged_string()))
            segment_text, segment_type = self._get_args_from_sentence(sentence)
            if segment_text and segment_type:
                self.app_logger.debug("Segment text: {}".format(segment_text))
                self.app_logger.debug("Segment type: {}".format(segment_type))
                segment_counter += 1
                try:
                    start_idx = document["content"].index(segment_text)
                except(Exception, BaseException):
                    try:
                        start_idx = document["content"].index(segment_text[:4])
                    except(Exception, BaseException):
                        start_idx = None
                if start_idx:
                    end_idx = start_idx + len(segment_text)
                else:
                    start_idx, end_idx = "", ""
                seg = {
                    "id": "T{}".format(segment_counter),
                    "type": segment_type,
                    "starts": str(start_idx),
                    "ends": str(end_idx),
                    "segment": segment_text
                }
                document["annotations"]["ADUs"].append(seg)
        return document

    @staticmethod
    def _get_adus(segments):
        major_claims, claims, premises = [], [], []
        for segment in segments:
            text = segment["segment"]
            segment_id = segment["id"]
            if segment["type"] == "major_claim":
                major_claims.append((text, segment_id))
            elif segment["type"] == "claim":
                claims.append((text, segment_id))
            else:
                premises.append((text, segment_id))
        return major_claims, claims, premises

    def _predict_relations(self, major_claims, claims, premises, json_obj):
        rel_counter = 0
        if major_claims and claims:
            for major_claim in major_claims:
                for claim in claims:
                    sentence_pair = "[CLS] " + claim[0] + " [SEP] " + major_claim[0]
                    self.app_logger.debug("Predicting relation for sentence pair: {}".format(sentence_pair))
                    sentence = Sentence(sentence_pair)
                    self.rel_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = self._get_label_with_max_conf(labels=labels)
                    if label and label != "other":
                        rel_counter += 1
                        rel_dict = {
                            "id": "R{}".format(rel_counter),
                            "type": label,
                            "arg1": claim[1],
                            "arg2": major_claim[1],
                            "confidence": conf
                        }
                        json_obj["annotations"]["Relations"].append(rel_dict)
        if claims and premises:
            for claim in claims:
                for premise in premises:
                    sentence_pair = "[CLS] " + premise[0] + " [SEP] " + claim[0]
                    self.app_logger.debug("Predicting relation for sentence pair: {}".format(sentence_pair))
                    sentence = Sentence(sentence_pair)
                    self.rel_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = self._get_label_with_max_conf(labels=labels)
                    if label and label != "other":
                        rel_counter += 1
                        rel_dict = {
                            "id": "R{}".format(rel_counter),
                            "type": label,
                            "arg1": premise[1],
                            "arg2": claim[1],
                            "confidence": conf
                        }
                        json_obj["annotations"]["Relations"].append(rel_dict)
        return json_obj

    def _predict_stance(self, major_claims, claims, json_obj):
        stance_counter = 0
        if major_claims and claims:
            for major_claim in major_claims:
                for claim in claims:
                    sentence_pair = "[CLS] " + claim[0] + " [SEP] " + major_claim[0]
                    self.app_logger.debug("Predicting stance for sentence pair: {}".format(sentence_pair))
                    sentence = Sentence(sentence_pair)
                    self.stance_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = self._get_label_with_max_conf(labels=labels)
                    if label and label != "other":
                        stance_counter += 1
                        stance_list = [{
                            "id": "A{}".format(stance_counter),
                            "type": label,
                            "confidence": conf
                        }]
                        for segment in json_obj["annotations"]["ADUs"]:
                            if segment["id"] == claim[1]:
                                segment["stance"] = stance_list
        return json_obj

    @staticmethod
    def _get_label_with_max_conf(labels: List[Label]):
        max_lbl, max_conf = "", 0.0
        if labels:
            for label in labels:
                lbl = label.value
                conf = label.score
                if conf > max_conf:
                    max_lbl = lbl
                    max_conf = conf
        return max_lbl, max_conf

    def _get_args_from_sentence(self, sentence: Sentence):
        tagged_string = sentence.to_tagged_string()
        tagged_string_split = tagged_string.split()
        words, labels = [], []
        for tok in tagged_string_split:
            if tok.startswith("<"):
                tok = tok.replace("<", "")
                tok = tok.replace(">", "")
                labels.append(tok)
            else:
                words.append(tok)
        self.app_logger.debug("Words and labels for current sentence: {}".format(words, labels))
        self.app_logger.debug("Extracting ADU from sentence...")
        idx = 0
        segment_text, segment_type = "", ""
        while idx < len(labels):
            label = labels[idx]
            self.app_logger.debug("Current label: {}".format(label))
            if label.startswith("B-"):
                segment_type = label.replace("B-", "")
                self.app_logger.debug("Found ADU with type: {}".format(segment_type))
                segment_text = words[idx]
                next_correct_label = "I-{}".format(segment_type)
                idx += 1
                if idx >= len(labels):
                    break
                next_label = labels[idx]
                while next_label == next_correct_label:
                    segment_text += " {}".format(words[idx])
                    idx += 1
                    if idx >= len(labels):
                        break
                    next_label = labels[idx]
            else:
                idx += 1
        return segment_text, segment_type
