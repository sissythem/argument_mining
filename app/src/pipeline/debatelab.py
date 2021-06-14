import json
from itertools import combinations

import numpy as np
import requests
from flair.data import Sentence

from src.pipeline.validation import JsonValidator
from src.training.models import SequentialModel, ClassificationModel, TopicModel, CustomAgglomerative
from src.training.preprocessing import CustomSentence
from src.utils import utils
from src.utils.config import AppConfig, Notification


class DebateLab:
    """
    Class representing the full pipeline of DebateLab
    """

    def __init__(self, app_config):
        """
        Constructor to initialize a DebateLab run. Uses the AppConfig class to access the properties & model info

        Args
            app_config (AppConfig): object with the project's configuration
        """
        self.app_config: AppConfig = app_config
        self.notification: Notification = Notification(app_config=app_config)
        self.app_logger = app_config.app_logger

        # load ADU model
        self.adu_model = SequentialModel(app_config=self.app_config, model_name="adu")
        self.adu_model.load()

        # load Relations model
        self.rel_model = ClassificationModel(app_config=self.app_config, model_name="rel")
        self.rel_model.load()

        # load Stance model
        self.stance_model = ClassificationModel(app_config=self.app_config, model_name="stance")
        self.stance_model.load()

        # initialize TopicModel
        self.topic_model = TopicModel(app_config=self.app_config)

        # initialize Clustering model
        self.clustering = CustomAgglomerative(app_config=app_config)

    def run_pipeline(self, notify=True):
        """
        Function to execute the DebateLab pipeline.

            | 1. Loads the documents from the SocialObservatory Elasticsearch
            | 2. Runs the argument mining & cross-document clustering pipelines
            | 3. Validates the json output objects & performs corrections
            | 4. Stores the results in the DebateLab Elasticsearch
            | 4. Notifies ICS
        """
        # TODO retrieve previous day's articles, save now to properties --> retrieve_kind="date"
        # retrive new documents
        documents = self.app_config.elastic_retrieve.retrieve_documents()
        # run Argument Mining pipeline
        documents, document_ids = self.run_argument_mining(documents=documents)
        self.app_logger.info(f"Valid document ids: {document_ids}")
        if notify:
            self.notification.notify_ics(ids_list=document_ids)
        # run cross-document clustering
        # relations_ids = self.run_clustering(documents=documents, document_ids=document_ids)
        relations, relation_ids = self.run_manual_clustering(documents=documents, document_ids=document_ids)
        if notify:
            self.notification.notify_ics(ids_list=relation_ids, kind="clustering")
        self.app_logger.info("Evaluation is finished!")

    # ************************** Classification ********************************************************
    def run_argument_mining(self, documents, export_schema=False, save=True):
        """
        Argument Mining pipeline:
        | 1. Predict ADUs for each document
        | 2. Predict relations & stance on the extracted ADUs
        | 3. Validate output json objects
        | 4. Perform corrections wherever necessary
        | 5. Save outputs in the DebateLab Elasticsearch

        Args
            | documents (list): list of documents extracted from the SocialObservatory Elasticsearch
            | export_schema (bool): True/False to extract json schema from the output json objects

        Returns
            tuple: the list of documents and the list of document ids of the **valid** json objects
        """
        document_ids = []
        invalid_document_ids = []
        corrected_ids = []
        validator = JsonValidator(app_config=self.app_config)
        for idx, document in enumerate(documents):
            document, segment_counter, rel_counter, stance_counter = self.predict(document=document)
            counters = {"adu": segment_counter, "rel": rel_counter, "stance": stance_counter}
            validation_errors, invalid_adus, corrected = validator.run_validation(document=document, counters=counters)
            if corrected:
                corrected_ids.append(document_ids)
            if not validation_errors and save:
                self.app_config.elastic_save.save_document(document=document)
                document_ids.append(document["id"])
            else:
                invalid_document_ids.append(document["id"])
                validator.save_invalid_json(document=document, validation_errors=validation_errors,
                                            invalid_adus=invalid_adus)
        validator.print_validation_results(document_ids, corrected_ids, invalid_document_ids)
        if export_schema:
            validator.export_json_schema(document_ids=document_ids)
        return documents, document_ids

    def predict(self, document):
        """
        Function to generate the json output file.
        | 1. Generates topics for the document
        | 2. Retrieves named entities
        | 3. Predicts the ADUs in the document
        | 4. Predicts relations & stance for the ADUs found in the previous step
        | 5. Returns the json updated with the generated information

        Args
            document (dict): the json with information on a specific document

        Returns
            tuple: the generated document and the counters for the produced ids of ADUs, relations and stance
        """
        self.app_logger.info(f"Extracting topics for document with title: {document['title']}")
        document["topics"] = self.topic_model.get_topics(content=document["content"])
        self.app_logger.info("Extracting named entities")
        entities = self._get_named_entities(doc_id=document["id"], content=document["content"])
        self.app_logger.info("Predicting ADUs from document")
        segments, segment_counter = self._predict_adus(document=document)
        # assumes major_claim label should appear once -- concatenation due to sentence splitting
        major_claims, claims, premises = utils.get_adus(segments)
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

    def _get_named_entities(self, doc_id, content):
        """
        Retrieves Named Entities using a RESTful API service

        Args
            doc_id (str): the id of the document
            content (str): the content of the document

        Returns
            list: the list of the entities found in the document
        """
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
        """
        ADU prediction pipeline. Uses ellogon to tokenize the content of a document and for each sentence, it uses
        the ADU trained model to predict the segments. Afterwards, for each sentence, the segments are extracted into
        a string and the document ADUs field is updated

        Args
            document (dict): a specific document from the documents list

        Returns
            tuple: the list of the ADUs and the counter used for their ids
        """
        # init document id & annotations
        adus = []
        self.app_logger.debug(f"Processing document with id: {document['id']} and name: {document}")
        segment_counter = 0
        sentences = utils.tokenize_with_spans(text=document["content"])
        found_mc = False
        for sentence in sentences:
            self.app_logger.debug(f"Predicting labels for sentence: {sentence}")
            sentence = CustomSentence(sentence)
            self.adu_model.model.predict(sentence, all_tag_prob=True)
            self.app_logger.debug(f"Output: {sentence.to_tagged_string()}")
            segments = utils.get_args_from_sentence(sentence)
            if segments:
                for segment in segments:
                    if len(segment["text"]) == 1:
                        continue
                    if segment["text"] and segment["label"]:
                        tokens = segment["text"].split()
                        if len(tokens) <= 2:
                            continue
                        if segment["label"] == "major_claim":
                            found_mc = True
                        self.app_logger.debug(f"Segment text: {segment['text']}")
                        self.app_logger.debug(f"Segment type: {segment['label']}")
                        segment_counter += 1
                        start_idx, end_idx = utils.find_segment_in_text(segment=segment, sentence=sentence)
                        if start_idx == -1 and end_idx == -1:
                            continue
                        seg = {
                            "id": f"T{segment_counter}",
                            "type": segment["label"],
                            "starts": str(start_idx),
                            "ends": str(end_idx),
                            "segment": segment["text"],
                            "confidence": segment["mean_conf"]
                        }
                        adus.append(seg)
        return self._check_major_claim(adus=adus, title=document["title"], mc_exists=found_mc,
                                       segment_counter=segment_counter)

    def _check_major_claim(self, adus, title, mc_exists, segment_counter):
        if not mc_exists:
            return self._handle_missing_major_claim(adus, title)
        else:
            major_claims = []
            for adu in adus:
                if adu["type"] == "major_claim":
                    major_claims.append(adu)
            if len(major_claims) == 1:
                return adus, segment_counter
            return self._handle_multiple_major_claims(major_claims=major_claims, adus=adus)

    def _handle_multiple_major_claims(self, major_claims, adus):
        major_claim = {"id": "T1", "type": "major_claim", "segment": ""}
        confs = []
        for idx, mc in enumerate(major_claims):
            if idx == 0:
                major_claim["starts"] = mc["starts"]
            else:
                if int(mc["starts"]) > int(major_claim["ends"]) + 5:
                    break
            confs.append(mc["confidence"])
            major_claim["ends"] = mc["ends"]
            major_claim["segment"] += " " + mc["segment"]
        major_claim["confidence"] = np.mean(confs)
        major_claim["segment"] = utils.replace_multiple_spaces_with_single_space(text=major_claim["segment"])
        return self._insert_major_claim(adus=adus, seg=major_claim)

    def _handle_missing_major_claim(self, adus, title):
        seg = {
            "id": "T1",
            "type": "major_claim",
            "starts": 0,
            "ends": len(title),
            "segment": title,
            "confidence": 0.99
        }
        return self._insert_major_claim(adus, seg)

    def _insert_major_claim(self, adus, seg):
        segment_counter = 2
        for adu in adus:
            adu["id"] = f"T{segment_counter}"
            segment_counter += 1
        adus.insert(0, seg)
        return adus, segment_counter

    def _predict_relations(self, major_claims, claims, premises):
        """
        Relations prediction pipeline

        Args
            | major_claims (list): a list of the major claims predicted in the previous step
            | claims (list): a list of the claims predicted in the previous step
            | premises (list): a list of the premises predicted in the previous step

        Returns
            tuple: the list of the predicted relations and the counter used to produce ids for the relations
        """
        rel_counter = 0
        relations = []
        if major_claims and claims:
            relations, rel_counter = self._get_relations(source=claims, target=major_claims, counter=rel_counter)
        if claims and premises:
            rel, rel_counter = self._get_relations(source=premises, target=claims, counter=rel_counter,
                                                   modify_source_list=True)
            relations += rel
        return relations, rel_counter

    def _get_relations(self, source, target, counter, modify_source_list=False):
        """
        Performs combinations creating pairs of source/target ADUs to predict their relations. If the predicted label
        is either support or attack, the relations list is updated with a new entry.

        Args
            | source (list): list of source ADUs
            | target (list): list of target ADUs
            | counter (int): counter to be used in the new relations' ids

        Returns
            tuple: the list of the predicted relations and the counter with the new value
        """
        initial_source = source
        relations = []
        already_predicted = []
        for adu2 in target:
            if modify_source_list:
                source = self._modify_source_adus(adu2, already_predicted, initial_source)
            for adu1 in source:
                sentence_pair = f"[CLS] {adu1[0]} [SEP] {adu2[0]}"
                self.app_logger.debug(f"Predicting relation for sentence pair: {sentence_pair}")
                sentence = Sentence(sentence_pair)
                self.rel_model.model.predict(sentence)
                labels = sentence.get_labels()
                label, conf = utils.get_label_with_max_conf(labels=labels)
                if label and label != "other":
                    counter += 1
                    rel_dict = {
                        "id": f"R{counter}",
                        "type": label,
                        "arg1": adu1[1],
                        "arg2": adu2[1],
                        "confidence": conf
                    }
                    relations.append(rel_dict)
                    already_predicted.append(adu1)
        return relations, counter

    def _modify_source_adus(self, adu2, already_predicted, source):
        adu2_start = adu2["starts"]
        adu2_end = adu2["ends"]
        source = self._remove_already_predicted(source=source, already_predicted=already_predicted)
        source = self._keep_k_closest(source=source, target_start=adu2_start, target_end=adu2_end)
        return source

    def _remove_already_predicted(self, source, already_predicted):
        if source and already_predicted:
            final_source = []
            for s in source:
                found = False
                for pred in already_predicted:
                    if pred["id"] == s["id"]:
                        found = True
                        break
                if not found:
                    final_source.append(s)
            return final_source
        return source

    def _keep_k_closest(self, source, target_start, target_end, k=5):
        source = sorted(source, key=lambda k: k['starts'], reverse=False)
        if type(target_start) == str:
            target_start = int(target_start)
        if type(target_end) == str:
            target_end = int(target_end)
        for s in source:
            s["distance_from_start"] = abs(target_start - s["ends"])
            s["distance_from_end"] = abs(target_end - s["starts"])
        source_from_start = sorted(source, key=lambda key: key['distance_from_start'], reverse=False)
        source_from_end = sorted(source, key=lambda key: key['distance_from_end'], reverse=False)
        final_source = source_from_start[:k] + source_from_end[:k]
        return final_source

    def _predict_stance(self, major_claims, claims, json_obj):
        """
        Stance prediction pipeline. It produces pairs of claims/major claims to predict their stance.

        Args
            | major_claims (list): list of the major claims of the document
            | claims (list): list of the claims of the document
            | json_obj (dict): the document

        Returns
            tuple: the updated document and the stance counter used to create ids for each stance
        """
        stance_counter = 0
        if major_claims and claims:
            for major_claim in major_claims:
                for claim in claims:
                    sentence_pair = "[CLS] " + claim[0] + " [SEP] " + major_claim[0]
                    self.app_logger.debug(f"Predicting stance for sentence pair: {sentence_pair}")
                    sentence = Sentence(sentence_pair)
                    self.stance_model.model.predict(sentence)
                    labels = sentence.get_labels()
                    label, conf = utils.get_label_with_max_conf(labels=labels)
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

    # ************************************* Cross-document relations **********************************
    def run_manual_clustering(self, documents, document_ids, save=True):
        self.app_logger.info("Running manual clustering -- agglomerative")
        adus, doc_ids, adu_ids = utils.collect_adu_for_clustering(documents=documents, document_ids=document_ids)
        self.app_logger.info("Collected claims for clustering")
        data_pairs = self.clustering.get_clusters(sentences=adus, doc_ids=doc_ids, sentences_ids=adu_ids)
        relations, relation_ids = [], []
        for pair in data_pairs:
            relation_id = f"{pair['doc_id1']};{pair['doc_id2']};{pair['sentence1_id']};{pair['sentence2_id']}"
            self.app_logger.debug(f"Saving cross document relation with id:{relation_id}")
            relation = {
                "id": relation_id,
                "cluster": pair['cluster'],
                "source": pair["sentence1_id"],
                "source_doc": pair["doc_id1"],
                "source_segment": pair["sentence1"],
                "target": pair["sentence2_id"],
                "target_doc": pair["doc_id2"],
                "target_segment": pair["sentence2"],
                "type": pair["type"],
                "score": pair["score"]
            }
            if save:
                self.app_config.elastic_save.save_relation(relation=relation)
            relations.append(relation)
            relation_ids.append(relation_id)
        return relations, relation_ids

    def run_clustering(self, documents, document_ids, save=False):
        """
        Cross-document relations pipeline

        Args
            | documents (list): list of json documents extracted from the SocialObservatory Elasticsearch & updated by
            the Argument Mining pipeline
            | document_ids (list): list of the ids of the valid documents
        """
        adus, doc_ids, adu_ids = utils.collect_adu_for_clustering(documents=documents, document_ids=document_ids)
        n_clusters = self.app_config.properties["clustering"]["n_clusters"]
        clusters = self.clustering.get_clusters(n_clusters=n_clusters, sentences=adus)
        relations = self.get_cross_document_relations(clusters=clusters, sentences=adus, adu_ids=adu_ids,
                                                      doc_ids=doc_ids)
        relations_ids = []
        for relation in relations:
            if save:
                self.app_config.elastic_save.save_relation(relation=relation)
            relations_ids.append(relation["id"])
        return relations_ids

    def get_cross_document_relations(self, clusters, sentences, adu_ids, doc_ids):
        cluster_dict = self.get_content_per_cluster(clusters=clusters, sentences=sentences, adu_ids=adu_ids,
                                                    doc_ids=doc_ids)
        relations = []
        for cluster, pairs in cluster_dict.items():
            cluster_combinations = list(combinations(pairs, r=2))
            for pair_combination in cluster_combinations:
                arg1 = pair_combination[0]
                arg2 = pair_combination[1]
                relation = {
                    "id": f"{arg1[1]};{arg2[1]};{arg1[0]};{arg2[0]}",
                    "cluster": cluster,
                    "source": arg1[0],
                    "source_doc": arg1[1],
                    "target": arg2[0],
                    "target_doc": arg2[1],
                    "type": "similar",
                    "score": 0.0
                }
                relations.append(relation)
        return relations

    def get_content_per_cluster(self, clusters, sentences, doc_ids, adu_ids, print_clusters=True):
        clusters_dict = {}
        clusters = clusters.labels_
        self.app_logger.debug(f"Clusters: {np.unique(clusters)}")
        for idx, cluster in enumerate(clusters):
            if cluster not in clusters_dict.keys():
                clusters_dict[cluster] = []
            adu_id = adu_ids[idx]
            doc_id = doc_ids[idx]
            sentence = sentences[idx]
            clusters_dict[cluster].append((adu_id, doc_id, sentence))
        if print_clusters:
            for idx, cluster_list in clusters_dict.items():
                self.app_logger.debug(f"Content of Cluster {idx}")
                for pair in cluster_list:
                    self.app_logger.debug(f"Sentence {pair[0]} in document with id {pair[1]}")
                    self.app_logger.debug(f"Sentence content: {pair[2]}")
        return clusters_dict