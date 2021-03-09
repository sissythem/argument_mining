import json
from itertools import combinations

import requests
from flair.data import Sentence

from pipeline.validation import JsonValidator
from training.models import SequentialModel, ClassificationModel, TopicModel, Clustering
from utils.config import AppConfig
from utils.utils import Utilities


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
        self.app_logger = app_config.app_logger
        self.utilities = Utilities(app_config=self.app_config)

        # load ADU model
        self.adu_model = SequentialModel(app_config=self.app_config, model_name="adu")
        self.adu_model.load()

        # load Relations model
        self.rel_model = ClassificationModel(app_config=self.app_config, model_name="rel")
        self.rel_model.load()

        # load Stance model
        self.stance_model = ClassificationModel(app_config=self.app_config, model_name="stance")
        self.stance_model.load()

    def run_pipeline(self):
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
        # update yml properties file with the latest retrieve date
        self.app_config.update_last_retrieve_date()
        # run Argument Mining pipeline
        documents, document_ids = self.run_argument_mining(documents=documents)
        # run cross-document clustering
        self.run_clustering(documents=documents, document_ids=document_ids)
        self.app_logger.info("Evaluation is finished!")

    # ************************** Classification ********************************************************
    def run_argument_mining(self, documents, export_schema=False):
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
            if not validation_errors:
                # TODO uncomment
                # self.app_config.elastic_save.save_document(document=document)
                document_ids.append(document["id"])
            else:
                invalid_document_ids.append(document["id"])
                self.utilities.save_invalid_json(document=document, validation_errors=validation_errors,
                                                 invalid_adus=invalid_adus)
        self.utilities.print_validation_results(document_ids, corrected_ids, invalid_document_ids)
        if export_schema:
            validator.export_json_schema(document_ids=document_ids)
        # TODO uncomment
        # self.app_config.notify_ics(ids_list=document_ids)
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
        topic_model = TopicModel(app_config=self.app_config)
        document["topics"] = topic_model.get_topics(content=document["content"])
        self.app_logger.info("Extracting named entities")
        entities = self._get_named_entities(doc_id=document["id"], content=document["content"])
        self.app_logger.info("Predicting ADUs from document")
        segments, segment_counter = self._predict_adus(document=document)
        # assumes major_claim label should appear once -- concatenation due to sentence splitting
        segments = self.utilities.concat_major_claim(segments=segments, title=document["title"],
                                                     content=document["content"], counter=segment_counter)
        major_claims, claims, premises = self.utilities.get_adus(segments)
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
        sentences = self.utilities.tokenize(text=document["content"], punct=False)
        for sentence in sentences:
            self.app_logger.debug(f"Predicting labels for sentence: {sentence}")
            sentence = " ".join(list(sentence))
            sentence = Sentence(sentence)
            self.adu_model.model.predict(sentence, all_tag_prob=True)
            self.app_logger.debug(f"Output: {sentence.to_tagged_string()}")
            segments = self.utilities.get_args_from_sentence(sentence)
            if segments:
                for segment in segments:
                    if segment["text"] and segment["label"]:
                        self.app_logger.debug(f"Segment text: {segment['text']}")
                        self.app_logger.debug(f"Segment type: {segment['label']}")
                        segment_counter += 1
                        start_idx, end_idx = self.utilities.find_segment_in_text(content=document["content"],
                                                                                 text=segment["text"])
                        seg = {
                            "id": f"T{segment_counter}",
                            "type": segment["label"],
                            "starts": str(start_idx),
                            "ends": str(end_idx),
                            "segment": segment["text"],
                            "confidence": segment["mean_conf"]
                        }
                        adus.append(seg)
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
            rel, rel_counter = self._get_relations(source=premises, target=claims, counter=rel_counter)
            relations += rel
        return relations, rel_counter

    def _get_relations(self, source, target, counter):
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
        relations = []
        for adu2 in target:
            for adu1 in source:
                sentence_pair = f"[CLS] {adu1[0]} [SEP] {adu2[0]}"
                self.app_logger.debug(f"Predicting relation for sentence pair: {sentence_pair}")
                sentence = Sentence(sentence_pair)
                self.rel_model.model.predict(sentence)
                labels = sentence.get_labels()
                label, conf = self.utilities.get_label_with_max_conf(labels=labels)
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
        return relations, counter

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
                    label, conf = self.utilities.get_label_with_max_conf(labels=labels)
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
    def run_clustering(self, documents, document_ids):
        """
        Cross-document relations pipeline

        Args
            | documents (list): list of json documents extracted from the SocialObservatory Elasticsearch & updated by
            the Argument Mining pipeline
            | document_ids (list): list of the ids of the valid documents
        """
        adus, doc_ids, adu_ids = self.utilities.collect_adu_for_clustering(documents=documents,
                                                                           document_ids=document_ids)
        clustering = Clustering(app_config=self.app_config)
        n_clusters = self.app_config.properties["clustering"]["n_clusters"]
        clusters = clustering.get_clusters(n_clusters=n_clusters, sentences=adus)
        relations = self.get_cross_document_relations(clusters=clusters, sentences=adus, adu_ids=adu_ids,
                                                      doc_ids=doc_ids)
        relations_ids = []
        for relation in relations:
            # TODO uncomment save to Elasticsearch -- change index inside the save_relation function!!
            # self.app_config.elastic_save.save_relation(relation=relation)
            relations_ids.append(relation["id"])
        # TODO notify ICS
        # self.app_config.notify_ics(ids_list=relations_ids, kind="clustering")

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
                    "target_doc": arg2[1]
                }
                relations.append(relation)
        return relations

    def get_content_per_cluster(self, clusters, sentences, doc_ids, adu_ids, print_clusters=True):
        clusters_dict = {}
        for idx, cluster in enumerate(clusters.labels_):
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
