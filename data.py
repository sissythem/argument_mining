import json
import pickle
import re
from os.path import join, exists
from typing import List
import pandas as pd
from ellogon import tokeniser


class Document:

    def __init__(self, app_logger, document_id, name, content, annotations):
        self.app_logger = app_logger
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
            self.app_logger.debug("Processing segment: {}".format(segment.text))
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
                tmp_txt = tmp_txt[:segment.char_start] + repl_char * \
                          len(segment.text) + tmp_txt[segment.char_end:]
            else:
                tmp_txt = tmp_txt.replace(
                    segment.text, repl_char * len(segment.text), 1)
        arg_counter = 0
        segments = []
        i = 0
        while i < len(tmp_txt):
            if tmp_txt[i] == repl_char:
                self.app_logger.debug("Found argument: {}".format(self.segments[arg_counter].text))
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
                self.app_logger.debug("Creating non argument segment for phrase: {}".format(rem))
                segment = Segment(segment_id=i, document_id=self.document_id, text=rem, char_start=start,
                                  char_end=end, arg_type=other_label)
                segment.sentences = tokeniser.tokenise_no_punc(segment.text)
                segment.bio_tagging(other_label=other_label)
                segments.append(segment)
        self.segments = segments

    def update_document(self):
        self.app_logger.debug("Update document sentences with labels")
        self.app_logger.debug("Updating document with id {}".format(self.document_id))
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
        self.app_logger.debug("All labels: {}".format(len(segment_labels)))
        label_idx = 0
        for s in self.sentences:
            self.app_logger.debug("Processing sentence: {}".format(s))
            sentence = []
            labels = []
            for token in s:
                if token:
                    sentence.append(token)
                    labels.append(segment_labels[label_idx])
                    label_idx += 1
            self.app_logger.debug("Labels for sentence: {}".format(labels))
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
                            sentence_labels.append(
                                "B-{}".format(self.arg_type))
                        else:
                            sentence_labels.append(
                                "I-{}".format(self.arg_type))
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
        self.resources_folder = app_config.resources_path
        self.pickle_file = app_config.documents_pickle
        self.load()

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
            self.app_logger.debug("Processing document with id: {}".format(document.document_id))
            doc_sentence_counter = 0
            for idx, sentence in enumerate(document.sentences):
                self.app_logger.debug("Processing sentence: {}".format(sentence))
                labels = document.sentences_labels[idx]
                for token, label in zip(sentence, labels):
                    is_arg = "Y" if label != "O" else "N"
                    sp = "SP: {}".format(doc_sentence_counter)
                    sentence_counter_str = "Sentence: {}".format(sentence_counter)
                    document_str = "Doc: {}".format(document.document_id)
                    df.loc[row_counter] = [token, label, is_arg, sp, sentence_counter_str, document_str]
                    row_counter += 1
                    sentence_counter += 1
                    doc_sentence_counter += 1
                df.loc[row_counter] = ["", "", "", "", "", ""]
                row_counter += 1
        self.app_logger.debug("Finished building dataframe. Saving...")
        out_file_path = join(resources, "train_adu.csv")
        df.to_csv(out_file_path, sep='\t', index=False, header=None)
        self.app_logger.debug("Dataframe saved!")

    def load_relations(self):
        relations, stances = self._get_relations()
        self._save_rel_df(rel_list=relations, filename=self.app_config.rel_train_csv)
        self._save_rel_df(rel_list=stances, filename=self.app_config.stance_train_csv)

    def _get_relations(self):
        resources = self.app_config.resources_path
        documents_path = join(resources, self.app_config.documents_pickle)
        self.app_logger.debug("Loading documents from pickle file")
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
        self.app_logger.debug("Documents are loaded")
        relations, stances = [], []
        for document in documents:
            self.app_logger.debug("Processing relations for document: {}".format(document.document_id))
            major_claims, claims, premises, relation_pairs, stance_pairs = self._collect_segments(document)
            relations += self._collect_relation_pairs(parents=major_claims, children=claims,
                                                      relation_pairs=relation_pairs)
            relations += self._collect_relation_pairs(parents=claims, children=premises, relation_pairs=relation_pairs)
            stances += self._collect_relation_pairs(parents=major_claims, children=claims, relation_pairs=stance_pairs)
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
            self.app_logger.debug("Text 1: {}".format(text1))
            self.app_logger.debug("Text 2: {}".format(text2))
            self.app_logger.debug("Pair label: {}".format(relation))
            final_text = "[CLS] " + text1 + " [SEP] " + text2
            sentence_counter_str = "Pair: {}".format(sentence_counter)
            df.loc[row_counter] = [final_text, relation, sentence_counter_str]
            row_counter += 1
            sentence_counter += 1
        output_filepath = join(resources_path, filename)
        df.to_csv(output_filepath, sep='\t', index=False, header=None)
        self.app_logger.debug("Dataframe saved!")

    def _collect_segments(self, document):
        major_claims, claims, premises = {}, {}, {}
        relation_pairs, stance_pairs = {}, {}
        relations: List[Relation] = document.relations
        stances = document.stance
        for relation in relations:
            if relation.arg1 is None or relation.arg2 is None:
                self.app_logger.error("None segment for relation: {} and document {}".format(relation.relation_id,
                                                                                             relation.document_id))
            relation_pairs[(relation.arg1.segment_id, relation.arg2.segment_id)] = relation.relation_type
        for stance in stances:
            if stance.arg1 is None or stance.arg2 is None:
                self.app_logger.error("None segment for relation: {} and document {}".format(stance.relation_id,
                                                                                             stance.document_id))
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

    def _collect_relation_pairs(self, parents, children, relation_pairs):
        new_relation_pairs = []
        count_relations = 0
        for p_id, p_text in parents.items():
            for c_id, c_text in children.items():
                key = (c_id, p_id)
                if key in relation_pairs.keys():
                    count_relations += 1
                relation = relation_pairs.get(key, "other")
                new_relation_pairs.append((c_text, p_text, relation))
        self.app_logger.debug("Found {} relations".format(count_relations))
        return new_relation_pairs

    def _create_document(self, doc):
        document = Document(app_logger=self.app_logger, document_id=doc["id"], name=doc["name"],
                            content=doc["text"], annotations=doc["annotations"])
        document.sentences = tokeniser.tokenise_no_punc(document.content)
        self.app_logger.debug(
            "Processing document with id {}".format(document.document_id))
        for annotation in document.annotations:
            annotation_id = annotation["_id"]
            spans = annotation["spans"]
            segment_type = annotation["type"]
            attributes = annotation["attributes"]
            if self._is_old_annotation(attributes):
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

    def _create_segment(self, span, document_id, annotation_id, attributes):
        segment_text = span["segment"]
        segment = Segment(segment_id=annotation_id, document_id=document_id, text=segment_text,
                          char_start=span["start"], char_end=span["end"], arg_type=attributes[0]["value"])
        segment.sentences = tokeniser.tokenise_no_punc(segment.text)
        segment.bio_tagging(other_label=self.app_config.properties["preprocessing"]["other_label"])
        return segment

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

    @staticmethod
    def _is_old_annotation(attributes):
        for attribute in attributes:
            name = attribute["name"]
            if name == "premise_type" or name == "premise" or name == "claim":
                return True
        return False
