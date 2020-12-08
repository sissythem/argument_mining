import pickle
from os.path import join, exists

import torch
from transformers import BertTokenizer
from base import utils


class DataPreprocessor:

    def __init__(self, app_config):
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.resources_folder = app_config.resources_path
        self.segments_pickle_file = app_config.pickle_segments_filename
        self.relations_file = app_config.relations_pickle_file
        self.stances_file = app_config.stances_pickle_file
        self.labels_to_int = {}
        self.int_to_labels = {}
        self.relations_to_int = {"other": 0, "support": 1, "attack": 2}
        self.int_to_relations = {0: "other", "support": 1, "attack": 2}
        self.stance_to_int = {"other": 0, "for": 1, "against": 2}
        self.int_to_stance = {0: "other", 1: "for", 2: "against"}
        self.tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/bert-base-greek-uncased-v1')

    def preprocess(self, documents):
        other_label = self.properties["preprocessing"]["other_label"]
        major_claims, claims, premises, relations, stances = self._collect_data(documents=documents,
                                                                                other_label=other_label)
        if exists(join(self.resources_folder, self.relations_file)):
            with open(join(self.resources_folder, self.relations_file), "rb") as f:
                relation_data, relation_labels, relation_initial_data, rel_lbls_dict = pickle.load(f)
        else:
            mc_data, mc_labels, mc_initial_data, rel_lbls_dict = self._collect_relations(segments1=major_claims,
                                                                                         segments2=claims,
                                                                                         relations=relations)
            cp_data, cp_labels, cp_initial_data, rel_lbls_dict = self._collect_relations(segments1=claims,
                                                                                         segments2=premises,
                                                                                         relations=relations)
            relation_data = mc_data + cp_data
            relation_labels = mc_labels + cp_labels
            relation_initial_data = mc_initial_data + cp_initial_data
            with open(join(self.resources_folder, self.relations_file), "wb") as f:
                pickle.dump((relation_data, relation_labels, relation_initial_data, rel_lbls_dict), f)

        if exists(join(self.resources_folder, self.stances_file)):
            with open(join(self.resources_folder, self.stances_file), "rb") as f:
                stance_data, stance_labels, stance_initial_data, lbls_dict = pickle.load(f)
        else:
            stance_data, stance_labels, stance_initial_data, lbls_dict = self._collect_relations(segments1=major_claims,
                                                                                                 segments2=claims,
                                                                                                 relations=stances,
                                                                                                 kind="stance")
            with open(join(self.resources_folder, self.stances_file), "wb") as f:
                pickle.dump((stance_data, stance_labels, stance_initial_data, lbls_dict), f)

        return {
            "relation": {
                "data": relation_data,
                "labels": relation_labels,
                "initial_data": relation_initial_data,
                "encoded_labels": rel_lbls_dict
            },
            "stance": {
                "data": stance_data,
                "labels": stance_labels,
                "initial_data": stance_initial_data,
                "encoded_labels": lbls_dict
            }
        }

    def _collect_data(self, documents, other_label="O"):
        relations, stances = {}, {}
        major_claims, claims, premises = [], [], []
        if exists(join(self.resources_folder, self.segments_pickle_file)):
            with open(join(self.resources_folder, self.segments_pickle_file), "rb") as f:
                major_claims, claims, premises, relations, stances = pickle.load(f)
        else:
            self.app_logger.debug("Processing documents to collect ADUs from all segments & their relations/stance")
            for document in documents:
                self.app_logger.debug("Find ADUs of document {}".format(document.document_id))
                for segment in document.segments:
                    self.app_logger.debug("Checking segment {}".format(segment.segment_id))
                    self.app_logger.debug("Segment is of type: {}".format(segment.arg_type))
                    tokens, labels = [], []
                    if segment.arg_type == other_label:
                        self.app_logger.debug("Segment is not ADU")
                        continue
                    for i, sentence in enumerate(segment.sentences):
                        for token in sentence:
                            if token:
                                tokens.append(token)
                                labels.append(segment.sentences_labels[i])
                    if tokens:
                        self.app_logger.debug("Collected tokens: {}".format(tokens))
                        segment.sentences = tokens
                        segment.sentences_labels = labels
                        segment.tokens = self._tokenize_sentence(segment)
                        segment.tokens, _ = utils.add_padding(segment.tokens)
                        self.app_logger.debug("Encoded data: {}".format(segment.tokens))
                        if "major" in segment.arg_type:
                            major_claims.append(segment)
                        elif "claim" in segment.arg_type:
                            claims.append(segment)
                        else:
                            premises.append(segment)
                self.app_logger.debug("Processing relations for document {}".format(document.document_id))
                for relation in document.relations:
                    arg1 = relation.arg1
                    arg2 = relation.arg2
                    self.app_logger.debug("Relation for argument1 {}".format(arg1.text))
                    self.app_logger.debug("Relation for argument2 {}".format(arg2.text))
                    self.app_logger.debug("Relation type : {}".format(relation.relation_type))
                    relations[(arg1.segment_id, arg2.segment_id)] = relation.relation_type
                for s in document.stance:
                    arg1 = s.arg1
                    arg2 = s.arg2
                    self.app_logger.debug("Stance for argument1 {}".format(arg1.text))
                    self.app_logger.debug("Stance for argument2 {}".format(arg2.text))
                    self.app_logger.debug("Stance type : {}".format(s.relation_type))
                    stances[(arg1.segment_id, arg2.segment_id)] = s.relation_type
            with open(join(self.resources_folder, self.segments_pickle_file), "wb") as f:
                pickle.dump((major_claims, claims, premises, relations, stances), f)
        return major_claims, claims, premises, relations, stances

    def _tokenize_sentence(self, segment):
        tokens = []
        for word in segment.sentences:
            tokenized = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            tokens.extend(tokenized)
        cls = self.tokenizer.cls_token_id
        sep = self.tokenizer.sep_token_id
        tokens.insert(0, cls)
        tokens.insert(len(tokens), sep)
        tokens = torch.LongTensor([tokens])
        return tokens

    def _collect_relations(self, segments1, segments2, relations, kind="relation"):
        encoded_labels = (self.relations_to_int, self.int_to_relations) if kind == "relation" else \
            (self.stance_to_int, self.int_to_stance)
        self.app_logger.debug("Collecting relations")
        array_size1 = len(segments1)
        array_size2 = len(segments2)
        data, labels, initial_data = [], [], []
        for i in range(array_size1):
            for j in range(array_size2):
                if i == j:
                    continue
                arg1 = segments1[i]
                arg2 = segments2[j]
                label = self.relations_to_int["other"] if kind == "relation" else self.stance_to_int["other"]
                try:
                    relation = relations[(arg1.segment_id, arg2.segment_id)]
                    label = self.relations_to_int[relation] if kind == "relation" else \
                        self.stance_to_int[relation]
                except(KeyError, Exception):
                    pass
                self.app_logger.debug("Segments relation: {}".format(label))
                input_data = (arg1.tokens, arg2.tokens)
                initial_data.append((arg1, arg2, label))
                data.append(input_data)
                labels.append(label)
        return data, labels, initial_data, encoded_labels
