import base64
import json
from datetime import datetime
from os.path import join
from typing import List

import numpy as np
import requests
import yaml
from elasticsearch_dsl import Search
from ellogon import tokeniser
from flair.data import Sentence, Label

from training.models import AduModel, RelationsModel, TopicModel, Clustering
from utils.config import AppConfig
from pipeline.validation import JsonValidator, JsonCorrector
from utils.utils import Utilities


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
        self.utilities = Utilities(app_config=self.app_config)

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
        client = self.app_config.elastic_retrieve.elasticsearch_client
        documents = self._retrieve(client=client)
        documents, document_ids, invalid_document_ids = self.run_argument_mining(documents=documents)
        self.run_clustering(documents=documents, document_ids=document_ids)
        # TODO uncomment notification
        # self.notify_ics(document_ids=document_ids)
        self.app_logger.info("Evaluation is finished!")

    def run_argument_mining(self, documents, export_schema=False):
        document_ids = []
        invalid_document_ids = []
        validator = JsonValidator(app_config=self.app_config)
        for idx, document in enumerate(documents):
            document, segment_counter, rel_counter, stance_counter = self.predict(document=document)
            validation_errors, invalid_adus = self.run_validation(validator=validator, document=document,
                                                                  segment_counter=segment_counter,
                                                                  rel_counter=rel_counter,
                                                                  stance_counter=stance_counter)
            if not validation_errors:
                self.app_config.elastic_save.elasticsearch_client.index(index='debatelab', ignore=400, refresh=True,
                                                                        doc_type='docket', id=document["id"],
                                                                        body=document)
                document_ids.append(document["id"])
            else:
                invalid_document_ids.append(document["id"])
                self.app_logger.debug("Writing invalid document into file")
                timestamp = datetime.now()
                filename = f"{document['id']}_{timestamp}.json"
                file_path = join(self.app_config.output_files, filename)
                with open(file_path, "w", encoding='utf8') as f:
                    f.write(json.dumps(document, indent=4, sort_keys=False, ensure_ascii=False))
                with open(f"{file_path}.txt", "w") as f:
                    for validation_error in validation_errors:
                        f.write(validation_error.value + "\n")
                    for invalid_adu in invalid_adus:
                        f.write(str(invalid_adu) + "\n")
        self.app_logger.info(f"Total valid documents: {len(document_ids)}")
        self.app_logger.info(f"Total invalid documents: {len(invalid_document_ids)}")
        self.app_logger.warn(f"Invalid document ids: {invalid_document_ids}")
        if export_schema:
            validator.export_json_schema(document_ids=document_ids)
        return documents, document_ids, invalid_document_ids

    def run_validation(self, validator, document, segment_counter, rel_counter, stance_counter, do_correction=False):
        validation_errors, invalid_adus = validator.validate(document=document)
        if do_correction and validation_errors:
            counter = self.app_config.properties["eval"]["max_correction_tries"]
            corrector = JsonCorrector(app_config=self.app_config, segment_counter=segment_counter,
                                      rel_counter=rel_counter, stance_counter=stance_counter)
            while counter > 0 and validation_errors:
                if not corrector.can_document_be_corrected(validation_errors=validation_errors):
                    break
                document = corrector.correction(document=document, invalid_adus=invalid_adus)
                validation_errors, invalid_adus = validator.validate(document=document)
                counter -= 1
        return validation_errors, invalid_adus

    def run_clustering(self, documents, document_ids):
        claims, doc_ids = [], []
        for document in documents:
            if document["id"] in document_ids:
                for adu in document["annotations"]["ADUs"]:
                    if adu["type"] == "claim":
                        claims.append(adu["segment"])
                        doc_ids.append(document["id"])
        clustering = Clustering(app_config=self.app_config)
        n_clusters = self.app_config.properties["clustering"]["n_clusters"]
        clusters = clustering.get_clusters(sentences=claims, n_clusters=n_clusters)
        clusters = list(clusters.labels_)
        clustering.get_content_per_cluster(n_clusters=n_clusters, clusters=clusters, sentences=claims, doc_ids=doc_ids)

    def _retrieve(self, client):
        retrieve_kind = self.app_config.properties["eval"]["retrieve"]
        if retrieve_kind == "file":
            file_path = join(self.app_config.resources_path, "kasteli_34_urls.txt")
            # read the list of urls from the file:
            with open(file_path, "r") as f:
                urls = [line.rstrip() for line in f]
            search_articles = Search(using=client, index='articles').filter('terms', link=urls)
        else:
            # TODO retrieve previous day's articles, save now to properties
            now = datetime.now()
            previous_date = self.app_config.properties["eval"]["last_date"]
            search_articles = Search(using=client, index='articles').filter('range',
                                                                            date={'gt': previous_date, 'lte': now})
            # update last search date
            properties = self.app_config.properties
            properties["eval"]["last_date"] = now
            with open(join(self.app_config.resources_path, self.app_config.properties_file), "w") as f:
                yaml.dump(properties, f)
        documents = []
        for hit in search_articles.scan():
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            if not document["content"].startswith(document["title"]):
                document["content"] = document["title"] + "\r\n\r\n" + document["content"]
            documents.append(document)
        return documents

    def notify_ics(self, document_ids):
        properties = self.app_config.properties["eval"]["notify"]
        url = properties["url"]
        username = properties["username"]
        password = properties["password"]
        data = {"properties": {"delivery_mode": 2}, "routing_key": "dlabqueue", "payload": json.dumps(document_ids),
                "payload_encoding": "string"}
        creds = f"{username}:{password}"
        creds_bytes = creds.encode("ascii")
        base64_bytes = base64.b64encode(creds_bytes)
        base64_msg = base64_bytes.decode("ascii")
        headers = {"Content-Type": "application/json", "Authorization": f"Basic {base64_msg}"}
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                self.app_logger.info("Request to ICS was successful!")
            else:
                self.app_logger.error(
                    f"Request to ICS failed with status code: {response.status_code} and message:{response.text}")
        except(BaseException, Exception) as e:
            self.app_logger.error(f"Request to ICS failed: {e}")

    def predict(self, document):
        self.app_logger.info(f"Extracting topics for document with title: {document['title']}")
        document["topics"] = self._get_topics(content=document["content"])
        self.app_logger.info("Extracting named entities")
        entities = self._get_named_entities(doc_id=document["id"], content=document["content"])
        self.app_logger.info("Predicting ADUs from document")
        segments, segment_counter = self._predict_adus(document=document)
        major_claims, claims, premises = self._get_adus(segments)
        self.app_logger.info(
            f"Found {len(major_claims)} major claims, {len(claims)} claims and {len(premises)} premises")
        self.app_logger.info("Predicting relations between ADUs")
        relations, rel_counter = self._predict_relations(major_claims=major_claims, claims=claims, premises=premises)
        document["annotations"] = {
            "ADUs": segments,
            "Relations": relations,
            "entities": entities
        }
        self.app_logger.info("Predicting stance between major claim and claims")
        document, stance_counter = self._predict_stance(major_claims=major_claims, claims=claims, json_obj=document)
        return document, segment_counter, rel_counter, stance_counter

    def _get_topics(self, content):
        sentences = tokeniser.tokenise(content)
        sentences = [" ".join(s) for s in sentences]
        self.app_logger.debug(f"Sentences fed to TopicModel: {sentences}")
        topic_model = TopicModel(app_config=self.app_config)
        topics = topic_model.get_topics(sentences=sentences)
        return topics

    def _get_named_entities(self, doc_id, content):
        entities = []
        data = {"text": content, "doc_id": doc_id}
        url = self.app_config.properties["eval"]["ner_endpoint"]
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

        self.app_logger.debug(f"Processing document with id: {document['id']} and name: {document}")
        segment_counter = 0
        sentences = tokeniser.tokenise_no_punc(document["content"])
        for sentence in sentences:
            self.app_logger.debug(f"Predicting labels for sentence: {sentence}")
            sentence = " ".join(list(sentence))
            sentence = Sentence(sentence)
            self.adu_model.model.predict(sentence, all_tag_prob=True)
            self.app_logger.debug(f"Output: {sentence.to_tagged_string()}")
            segments = self._get_args_from_sentence(sentence)
            segments = self._concat_major_claim(segments=segments, title=document["title"])
            if segments:
                for segment in segments:
                    if segment.text and segment.label:
                        self.app_logger.debug(f"Segment text: {segment.text}")
                        self.app_logger.debug(f"Segment type: {segment.label}")
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
                            "id": f"T{segment_counter}",
                            "type": segment.label,
                            "starts": str(start_idx),
                            "ends": str(end_idx),
                            "segment": segment.text,
                            "confidence": segment.mean_conf
                        }
                        adus.append(seg)
        return adus, segment_counter

    def _concat_major_claim(self, segments: List[Segment], title):
        if not segments:
            return []
        new_segments = []
        major_claim_txt = ""
        major_claims = [mc for mc in segments if mc.label == "major_claim"]
        mc_exists = False
        if major_claims:
            mc_exists = True
            for mc in major_claims:
                major_claim_txt += f" {mc.text}"
        else:
            major_claim_txt = title
        major_claim_txt = self.utilities.replace_multiple_spaces_with_single_space(text=major_claim_txt)
        already_found_mc = False
        for segment in segments:
            if not mc_exists and not already_found_mc:
                major_claim = ArgumentMining.Segment(text=major_claim_txt, label="major_claim")
                major_claim.mean_conf = 0.99
                new_segments.append(major_claim)
                already_found_mc = True
            if segment.label == "major_claim":
                if not already_found_mc:
                    segment.text = major_claim_txt
                    new_segments.append(segment)
                    already_found_mc = True
                else:
                    continue
            else:
                new_segments.append(segment)
        return new_segments

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
                    self.app_logger.debug(f"Predicting relation for sentence pair: {sentence_pair}")
                    sentence = Sentence(sentence_pair)
                    self.rel_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = self._get_label_with_max_conf(labels=labels)
                    if label and label != "other":
                        rel_counter += 1
                        rel_dict = {
                            "id": f"R{rel_counter}",
                            "type": label,
                            "arg1": claim[1],
                            "arg2": major_claim[1],
                            "confidence": conf
                        }
                        relations.append(rel_dict)
        if claims and premises:
            for claim in claims:
                for premise in premises:
                    sentence_pair = f"[CLS] {premise[0]} [SEP] {claim[0]}"
                    self.app_logger.debug(f"Predicting relation for sentence pair: {sentence_pair}")
                    sentence = Sentence(sentence_pair)
                    self.rel_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = self._get_label_with_max_conf(labels=labels)
                    if label and label != "other":
                        rel_counter += 1
                        rel_dict = {
                            "id": f"R{rel_counter}",
                            "type": label,
                            "arg1": premise[1],
                            "arg2": claim[1],
                            "confidence": conf
                        }
                        relations.append(rel_dict)
        return relations, rel_counter

    def _predict_stance(self, major_claims, claims, json_obj):
        stance_counter = 0
        if major_claims and claims:
            for major_claim in major_claims:
                for claim in claims:
                    sentence_pair = "[CLS] " + claim[0] + " [SEP] " + major_claim[0]
                    self.app_logger.debug(f"Predicting stance for sentence pair: {sentence_pair}")
                    sentence = Sentence(sentence_pair)
                    self.stance_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = self._get_label_with_max_conf(labels=labels)
                    if label and label != "other":
                        stance_counter += 1
                        stance_list = [{
                            "id": f"A{stance_counter}",
                            "type": label,
                            "confidence": conf
                        }]
                        for segment in json_obj["annotations"]["ADUs"]:
                            if segment["id"] == claim[1]:
                                segment["stance"] = stance_list
        return json_obj, stance_counter

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
                self.app_logger.debug(f"Returning completed segment: {segment.text}")
                return segment, current_idx
        else:
            # only care about B-tags to start a segment
            if label_type == "B":
                segment = ArgumentMining.Segment(text=token.text, label=label)
                segment.confidences.append(confidence)
                return self._get_next_segment(tokens, current_idx + 1, label, segment)
            else:
                return self._get_next_segment(tokens, current_idx + 1, None, segment)
