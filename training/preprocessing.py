import json
import pickle
import re
from os.path import join, exists
from typing import List

import pandas as pd

from utils.utils import Utilities


class Document:
    """
    Class representing a document. Contains the sentences, segments, annotations, relations etc
    """

    def __init__(self, app_config, document_id, name, content, annotations):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.utilities = Utilities(app_config=app_config)
        self.document_id: int = document_id
        self.name: str = name
        self.content: str = content
        self.annotations: List[dict] = annotations
        self.segments: List[Segment] = []
        self.relations: List[Relation] = []
        self.stance: List[Relation] = []
        self.sentences = []
        self.sentences_labels = []

    def update_segments(self, repl_char="=", other_label="O"):
        tmp_txt = self.content
        for segment in self.segments:
            self.app_logger.debug(f"Processing segment: {segment.text}")
            try:
                indices = [m.start() for m in re.finditer(segment.text, tmp_txt)]
            except (BaseException, Exception):
                indices = []
                pass
            if indices and len(indices) > 1:
                diffs = []
                for idx in indices:
                    diffs.append(abs(idx - segment.char_start))
                min_diff = min(diffs)
                idx_min = diffs.index(min_diff)
                segment.char_start = indices[idx_min]
                segment.char_end = segment.char_start + len(segment.text)
                tmp_txt = tmp_txt[:segment.char_start] + repl_char * len(segment.text) + tmp_txt[segment.char_end:]
            else:
                tmp_txt = tmp_txt.replace(
                    segment.text, repl_char * len(segment.text), 1)
        arg_counter = 0
        segments = []
        i = 0
        while i < len(tmp_txt):
            if tmp_txt[i] == repl_char:
                self.app_logger.debug(f"Found argument: {self.segments[arg_counter].text}")
                self.segments[arg_counter].char_start = i
                i += len(self.segments[arg_counter].text)
                self.segments[arg_counter].char_end = i
                segments.append(self.segments[arg_counter])
                arg_counter += 1
            else:
                rem = ""
                start = i
                while tmp_txt[i] != repl_char:
                    rem += tmp_txt[i]
                    i += 1
                    if i >= len(tmp_txt):
                        break
                end = i
                self.app_logger.debug(f"Creating non argument segment for phrase: {rem}")
                segment = Segment(segment_id=i, document_id=self.document_id, text=rem, char_start=start,
                                  char_end=end, arg_type=other_label)
                segment.sentences = self.utilities.tokenize(text=segment.text, punct=False)
                segment.bio_tagging(other_label=other_label)
                segments.append(segment)
        self.segments = segments

    def update_document(self):
        self.app_logger.debug("Update document sentences with labels")
        self.app_logger.debug(f"Updating document with id {self.document_id}")
        segment_tokens = []
        segment_labels = []
        sentences = []
        sentences_labels = []
        for segment in self.segments:
            for sentence in segment.sentences:
                for token in sentence:
                    segment_tokens.append(token)
            for labels in segment.sentences_labels:
                for label in labels:
                    segment_labels.append(label)
        self.app_logger.debug(f"All labels: {len(segment_labels)}")
        label_idx = 0
        for s in self.sentences:
            self.app_logger.debug(f"Processing sentence: {s}")
            sentence = []
            labels = []
            for token in s:
                if token:
                    sentence.append(token)
                    labels.append(segment_labels[label_idx])
                    label_idx += 1
            self.app_logger.debug(f"Labels for sentence: {labels}")
            if sentence:
                sentences.append(sentence)
                sentences_labels.append(labels)
        self.sentences = sentences
        self.sentences_labels = sentences_labels


class Segment:

    def __init__(self, segment_id, document_id, text, char_start, char_end, arg_type):
        self.segment_id: str = segment_id
        self.document_id: int = document_id
        self.text: str = text
        self.char_start: int = char_start
        self.char_end: int = char_end
        self.arg_type: str = arg_type
        self.sentences = []
        self.sentences_labels = []

    def bio_tagging(self, other_label="O"):
        sentences = []
        for sentence in self.sentences:
            sentence_labels = []
            tokens = []
            for token in sentence:
                if token:
                    tokens.append(token)
                    if self.arg_type == other_label:
                        sentence_labels.append(self.arg_type)
                    else:
                        if sentence.index(token) == 0:
                            sentence_labels.append(f"B-{self.arg_type}")
                        else:
                            sentence_labels.append(f"I-{self.arg_type}")
            sentences.append(tokens)
            self.sentences_labels.append(sentence_labels)
        self.sentences = sentences


class Relation:

    def __init__(self, relation_id, document_id, arg1, arg2, kind, relation_type):
        self.relation_id: str = relation_id
        self.document_id: int = document_id
        self.arg1: Segment = arg1
        self.arg2: Segment = arg2
        self.kind: str = kind
        self.relation_type: str = relation_type


class DataLoader:

    def __init__(self, app_config):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.utilities = Utilities(app_config=app_config)
        self.resources_folder = app_config.resources_path
        self.pickle_file = app_config.documents_pickle
        self.load()

    # **************************** Create document.pkl **********************************************
    def load(self, filename="kasteli.json"):
        path_to_pickle = join(self.resources_folder,
                              self.app_config.documents_pickle)
        if exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                return pickle.load(f)

        path_to_data = join(self.resources_folder, filename)
        with open(path_to_data, "r") as f:
            content = json.loads(f.read())
        documents = content["data"]["documents"]
        docs = []
        for doc in documents:
            document = self._create_document(doc)
            docs.append(document)
        with open(join(self.resources_folder, self.pickle_file), "wb") as f:
            pickle.dump(docs, f)
        return docs

    def _create_document(self, doc):
        document = Document(app_config=self.app_config, document_id=doc["id"], name=doc["name"],
                            content=doc["text"], annotations=doc["annotations"])
        document.sentences = self.utilities.tokenize(text=document.content, punct=False)
        self.app_logger.debug(
            f"Processing document with id {document.document_id}")
        for annotation in document.annotations:
            annotation_id = annotation["_id"]
            spans = annotation["spans"]
            segment_type = annotation["type"]
            attributes = annotation["attributes"]
            if self.utilities.is_old_annotation(attributes):
                continue
            if segment_type == "argument":
                span = spans[0]
                segment = self._create_segment(span=span, document_id=document.document_id,
                                                        annotation_id=annotation_id, attributes=attributes)
                document.segments.append(segment)
            elif segment_type == "argument_relation":
                relation = self._create_relation(segments=document.segments, attributes=attributes,
                                                          annotation_id=annotation_id, document_id=document.document_id)
                if relation.kind == "relation":
                    document.relations.append(relation)
                else:
                    document.stance.append(relation)
        document.segments.sort(key=lambda x: x.char_start)
        document.update_segments(repl_char=self.app_config.properties["preprocessing"]["repl_char"],
                                 other_label=self.app_config.properties["preprocessing"]["other_label"])
        document.update_document()
        document.segments.sort(key=lambda x: x.char_start)
        return document

    @staticmethod
    def _create_relation(segments, attributes, annotation_id, document_id):
        relation_type, kind, arg1_id, arg2_id = "", "", "", ""
        arg1, arg2 = None, None
        for attribute in attributes:
            name = attribute["name"]
            value = attribute["value"]
            if name == "type":
                relation_type = value
                if relation_type == "support" or relation_type == "attack":
                    kind = "relation"
                else:
                    kind = "stance"
            elif name == "arg1":
                arg1_id = value
            elif name == "arg2":
                arg2_id = value
        for seg in segments:
            if seg.segment_id == arg1_id:
                arg1 = seg
            elif seg.segment_id == arg2_id:
                arg2 = seg
        return Relation(relation_id=annotation_id, document_id=document_id, arg1=arg1,
                        arg2=arg2, kind=kind, relation_type=relation_type)

    def _create_segment(self, span, document_id, annotation_id, attributes):
        segment_text = span["segment"]
        segment = Segment(segment_id=annotation_id, document_id=document_id, text=segment_text,
                          char_start=span["start"], char_end=span["end"], arg_type=attributes[0]["value"])
        segment.sentences = self.utilities.tokenize(text=segment.text, punct=False)
        segment.bio_tagging(other_label=self.app_config.properties["preprocessing"]["other_label"])
        return segment

    # ********************************** Create ADUs csv file ****************************************
    def load_adus(self):
        self.app_logger.debug("Running ADU preprocessing")
        resources = self.app_config.resources_path
        documents_path = join(resources, self.app_config.documents_pickle)
        self.app_logger.debug("Loading documents from pickle file")
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
        self.app_logger.debug("Documents are loaded")
        df = pd.DataFrame(columns=["token", "label", "is_arg", "sp", "sentence", "document"])
        row_counter = 0
        sentence_counter = 0
        for document in documents:
            self.app_logger.debug(f"Processing document with id: {document.document_id}")
            doc_sentence_counter = 0
            for idx, sentence in enumerate(document.sentences):
                self.app_logger.debug(f"Processing sentence: {sentence}")
                labels = document.sentences_labels[idx]
                for token, label in zip(sentence, labels):
                    is_arg = "Y" if label != "O" else "N"
                    sp = f"SP: {doc_sentence_counter}"
                    sentence_counter_str = f"Sentence: {sentence_counter}"
                    document_str = f"Doc: {document.document_id}"
                    df.loc[row_counter] = [token, label, is_arg, sp, sentence_counter_str, document_str]
                    row_counter += 1
                    sentence_counter += 1
                    doc_sentence_counter += 1
                df.loc[row_counter] = ["", "", "", "", "", ""]
                row_counter += 1
        self.app_logger.debug("Finished building dataframe. Saving...")
        out_file_path = join(resources, "data", "train_adu.csv")
        df.to_csv(out_file_path, sep='\t', index=False, header=False)
        self.app_logger.debug("Dataframe saved!")

    # **************************** Create relations and stance csv files *******************************
    def load_relations(self, do_oversample=False):
        relations, stances = self._get_relations()
        self._save_rel_df(rel_list=relations, filename=self.app_config.rel_train_csv)
        self._save_rel_df(rel_list=stances, filename=self.app_config.stance_train_csv)
        if do_oversample:
            utility = Utilities(app_config=self.app_config)
            # TODO oversampling in a dynamic way
            utility.oversample(task_kind="rel", file_kind="train", total_num=8705)
            utility.oversample(task_kind="stance", file_kind="train", total_num=289)

    def _get_relations(self):
        resources = self.app_config.resources_path
        documents_path = join(resources, self.app_config.documents_pickle)
        self.app_logger.debug("Loading documents from pickle file")
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
        self.app_logger.debug("Documents are loaded")
        relations, stances = [], []
        for document in documents:
            self.app_logger.debug(f"Processing relations for document: {document.document_id}")
            major_claims, claims, premises, relation_pairs, stance_pairs = self._collect_segments(document)
            relations += self.utilities.collect_relation_pairs(parents=major_claims, children=claims,
                                                               relation_pairs=relation_pairs)
            relations += self.utilities.collect_relation_pairs(parents=claims, children=premises,
                                                               relation_pairs=relation_pairs)
            self.app_logger.debug(f"Found {len(relations)} relations")
            stances += self.utilities.collect_relation_pairs(parents=major_claims, children=claims,
                                                             relation_pairs=stance_pairs)
            self.app_logger.debug(f"Found {len(stances)} stance")
        return relations, stances

    def _save_rel_df(self, rel_list, filename):
        resources_path = self.app_config.resources_path
        df = pd.DataFrame(columns=["token", "label", "sentence"])
        row_counter = 0
        sentence_counter = 0
        for pair in rel_list:
            text1 = pair[0]
            text2 = pair[1]
            relation = pair[2]
            self.app_logger.debug("Processing pair:")
            self.app_logger.debug(f"Text 1: {text1}")
            self.app_logger.debug(f"Text 2: {text2}")
            self.app_logger.debug(f"Pair label: {relation}")
            final_text = f"[CLS] {text1} [SEP] {text2}"
            sentence_counter_str = f"Pair: {sentence_counter}"
            df.loc[row_counter] = [final_text, relation, sentence_counter_str]
            row_counter += 1
            sentence_counter += 1
        output_filepath = join(resources_path, "data", filename)
        df.to_csv(output_filepath, sep='\t', index=False, header=False)
        self.app_logger.debug("Dataframe saved!")

    def _collect_segments(self, document):
        major_claims, claims, premises = {}, {}, {}
        relation_pairs, stance_pairs = {}, {}
        relations: List[Relation] = document.relations
        stances = document.stance
        for relation in relations:
            if relation.arg1 is None or relation.arg2 is None:
                self.app_logger.error(
                    f"None segment for relation: {relation.relation_id} and document {relation.document_id}")
            relation_pairs[(relation.arg1.segment_id, relation.arg2.segment_id)] = relation.relation_type
        for stance in stances:
            if stance.arg1 is None or stance.arg2 is None:
                self.app_logger.error(
                    f"None segment for relation: {stance.relation_id} and document {stance.document_id}")
            stance_pairs[(stance.arg1.segment_id, stance.arg2.segment_id)] = stance.relation_type
        for segment in document.segments:
            if segment.arg_type == "major_claim":
                major_claims[segment.segment_id] = segment.text
            elif segment.arg_type == "claim":
                claims[segment.segment_id] = segment.text
            elif segment.arg_type == "premise":
                premises[segment.segment_id] = segment.text
            else:
                continue
        return major_claims, claims, premises, relation_pairs, stance_pairs
