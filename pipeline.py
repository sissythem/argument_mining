import json
import re
from os.path import join
from typing import List

import numpy as np
import requests
from elasticsearch_dsl import Search
from ellogon import tokeniser
from flair.data import Sentence, Label
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import utils
from classifiers import AduModel, RelationsModel
from utils import AppConfig


# import yaml


class ArgumentMining:
    class Segment:
        def __init__(self, text, label):
            self.text = text
            self.label = label
            self.confidences = []
            self.mean_conf = 0.0

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
                self.app_config.elastic_save.elasticsearch_client.index(index='debatelab', ignore=400, refresh=True,
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

    def _get_topics(self, content):
        n_features = 1000
        n_components = 400
        n_top_words = 10
        greek_stopwords = stopwords.words("greek")
        regex = re.compile(r'\b(' + greek_stopwords + ')\b', flags=re.IGNORECASE)
        processed_content = regex.sub("", content)
        self.app_logger.info("Extracting topics")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words=greek_stopwords)
        tf = tf_vectorizer.fit_transform([processed_content])
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit_transform(tf)
        feature_names = tf_vectorizer.get_feature_names()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topics += [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return topics

    def predict(self, document):
        entities = self._get_named_entities(doc_id=document["id"], content=document["content"])
        segments = self._predict_adus(document=document)
        major_claims, claims, premises = self._get_adus(segments)
        relations = self._predict_relations(major_claims=major_claims, claims=claims, premises=premises)
        document["annotations"] = {
            "ADUs": segments,
            "Relations": relations,
            "entities": entities
        }
        document["topics"] = self._get_topics(content=document["content"])
        json_obj = self._predict_stance(major_claims=major_claims, claims=claims, json_obj=document)
        return json_obj

    def _get_named_entities(self, doc_id, content):
        entities = []
        data = {"text": content, "doc_id": doc_id}
        url = self.app_config.properties["ner_endpoint"]
        response = requests.post(url, data=json.dumps(data))
        if response.status_code == 200:
            entities = json.loads(response.text)
        else:
            self.app_logger.error("Could not retrieve named entities")
            self.app_logger.error(f"Status code: {response.status_code}, reason: {response.text}")
        return entities

    def _predict_adus(self, document):
        # init document id & annotations
        adus = []

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
            segments: ArgumentMining.Segment = self._get_args_from_sentence(sentence)
            if segments:
                for segment in segments:
                    if segment.text and segment.label:
                        self.app_logger.debug("Segment text: {}".format(segment.text))
                        self.app_logger.debug("Segment type: {}".format(segment.label))
                        segment_counter += 1
                        try:
                            start_idx = document["content"].index(segment.text)
                        except(Exception, BaseException):
                            try:
                                start_idx = document["content"].index(segment.text[:10])
                            except(Exception, BaseException):
                                start_idx = None
                        if start_idx:
                            end_idx = start_idx + len(segment.text)
                        else:
                            start_idx, end_idx = "", ""
                        seg = {
                            "id": "T{}".format(segment_counter),
                            "type": segment.label,
                            "starts": str(start_idx),
                            "ends": str(end_idx),
                            "segment": segment.text,
                            "confidence": segment.mean_conf
                        }
                        adus.append(seg)
        return adus

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

    def _predict_relations(self, major_claims, claims, premises):
        rel_counter = 0
        relations = []
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
                        relations.append(rel_dict)
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
                        relations.append(rel_dict)
        return relations

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

    def _get_args_from_sentence(self, sentence: Sentence) -> List[Segment]:
        if sentence.tokens:
            segments = []
            idx = None
            while True:
                segment, idx = self._get_next_segment(sentence.tokens, current_idx=idx)
                if segment:
                    segment.mean_conf = np.mean(segment.confidences)
                    segments.append(segment)
                if idx is None:
                    break
            return segments

    def _get_next_segment(self, tokens, current_idx=None, current_label=None, segment: Segment = None):
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
                segment.text += " " + token.text
                segment.confidences.append(confidence)
                return self._get_next_segment(tokens, current_idx + 1, current_label, segment)
            else:
                # new segment, different than the current one
                # next function call should start at the current_idx
                self.app_logger.debug("Returning completed segment: {}".format(segment.text))
                return segment, current_idx
        else:
            # only care about B-tags to start a segment
            if label_type == "B":
                segment = ArgumentMining.Segment(text=token.text, label=label)
                segment.confidences.append(confidence)
                return self._get_next_segment(tokens, current_idx + 1, label, segment)
            else:
                return self._get_next_segment(tokens, current_idx + 1, None, segment)
