import json
import pickle
import re
from os.path import join, exists

import torch
from ellogon import tokeniser
from transformers import BertTokenizer

def tokenize_sentences(text):
    return tokeniser.tokenise_no_punc(text)

class Document:

    def __init__(self, app_logger, document_id, name, content, annotations):
        self.app_logger = app_logger
        self.document_id = document_id
        self.name = name
        self.content = content
        self.annotations = annotations
        self.segments = []
        self.relations = []
        self.stance = []
        self.sentences = []
        self.sentences_labels = []

    def update_segments(self, repl_char="=", other_label="O"):
        tmp_txt = self.content
        for segment in self.segments:
            self.app_logger.debug(
                "Processing segment: {}".format(segment.text))
            try:
                indices = [m.start()
                           for m in re.finditer(segment.text, tmp_txt)]
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
                self.app_logger.debug("Found argument: {}".format(
                    self.segments[arg_counter].text))
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
        self.segment_id = segment_id
        self.document_id = document_id
        self.text = text
        self.char_start = char_start
        self.char_end = char_end
        self.arg_type = arg_type
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
        self.relation_id = relation_id
        self.document_id = document_id
        self.arg1 = arg1
        self.arg2 = arg2
        self.kind = kind
        self.relation_type = relation_type


class DataLoader:

    def __init__(self, app_config):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.resources_folder = app_config.resources_path
        self.data_file = app_config.data_file
        self.pickle_file = app_config.documents_pickle

    def load_data(self):
        path_to_pickle = join(self.resources_folder,
                              self.app_config.documents_pickle)
        if exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                return pickle.load(f)

        path_to_data = join(self.resources_folder, self.data_file)
        with open(path_to_data, "r") as f:
            content = json.loads(f.read())
        documents = content["data"]["documents"]
        docs = []
        for doc in documents:
            # TODO remove the following if
            if doc["id"] == 204:
                continue
            document = self._create_document(doc)
            docs.append(document)
        with open(join(self.resources_folder, self.pickle_file), "wb") as f:
            pickle.dump(docs, f)
        return docs

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


class DataPreprocessor:

    def __init__(self, app_config):
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.resources_folder = app_config.resources_path
        self.pickle_file = app_config.pickle_sentences_filename
        self.segments_pickle_file = app_config.pickle_segments_filename
        self.relations_file = app_config.relations_pickle_file
        self.stances_file = app_config.stances_pickle_file
        self.relations_pickle = app_config.relations_pickle_file
        self.labels_to_int = {}
        self.int_to_labels = {}
        self.relations_to_int = {"other": 0, "support": 1, "attack": 2}
        self.int_to_relations = {0: "other", "support": 1, "attack": 2}
        self.stance_to_int = {"other": 0, "for": 1, "against": 2}
        self.int_to_stance = {0: "other", 1: "for", 2: "against"}
        self.tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/bert-base-greek-uncased-v1')

    def preprocess(self, documents):
        if exists(join(self.resources_folder, self.pickle_file)):
            with open(join(self.resources_folder, self.pickle_file), "rb") as f:
                return pickle.load(f)

        final_sentences, final_labels = [], []
        self.labels_to_int = self._collect_unique_labels(documents)
        for label, number in self.labels_to_int.items():
            self.int_to_labels[number] = label
        self.app_logger.debug("Labels to ints: {}".format(self.labels_to_int))
        self._transform_labels(documents)
        sentences, labels = self._collect_instances(documents=documents)
        for sentence, lbls in zip(sentences, labels):
            if len(sentence) != len(lbls):
                self.app_logger.error("Found sentence and labels with different lengths")
                self.app_logger.error("Sentence: {}".format(sentence))
                self.app_logger.error("Labels: {}".format(labels))
                break
            self.app_logger.debug(
                "Processing sentence {} and labels {}".format(sentence, lbls))
            tokens, new_labels = self._get_encoded_sentence_and_labels(sentence=sentence, labels=lbls,
                                                                       other_label=self.properties["preprocessing"][
                                                                           "other_label"])
            final_sentences.append(tokens)
            final_labels.append(new_labels)
        max_len = self.properties["preprocessing"]["max_len"]
        pad_token = self.properties["preprocessing"]["pad_token"]
        final_sentences, final_labels = self._add_padding(tokens=final_sentences, labels=final_labels, max_len=max_len,
                                                          pad_token=pad_token)
        output = (final_sentences, final_labels, self.labels_to_int)
        with open(join(self.resources_folder, self.pickle_file), "wb") as f:
            pickle.dump(obj=output, file=f)
        return output

    def preprocess_relations(self, documents):
        other_label = self.properties["preprocessing"]["other_label"]
        all_segments, stance_segments, relations, stances = self._collect_data(documents=documents,
                                                                               other_label=other_label)
        relation_data, relation_labels, relation_initial_data, rel_lbls_dict = (1,1,1,1)
        # relation_data, relation_labels, relation_initial_data, rel_lbls_dict = self._collect_relations(
        #     segments=all_segments,
        #     relations=relations,
        #     pickle_file=self.relations_file)
        stance_data, stance_labels, stance_initial_data, lbls_dict = self._collect_relations(segments=stance_segments,
                                                                                             relations=stances,
                                                                                             pickle_file=self.stances_file,
                                                                                             kind="stance")
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

    def _get_encoded_sentence_and_labels(self, sentence, labels, other_label="O"):
        tokens, new_labels = self._get_sentence_tokens_and_labels(
            sentence=sentence, labels=labels)
        self.app_logger.debug("New tokens {} and labels {}".format(tokens, new_labels))
        cls = self.tokenizer.cls_token_id
        sep = self.tokenizer.sep_token_id
        tokens.insert(0, cls)
        new_labels.insert(0, self.labels_to_int[other_label])
        tokens.insert(len(tokens), sep)
        new_labels.insert(len(new_labels), self.labels_to_int[other_label])
        tokens = torch.LongTensor([tokens])
        new_labels = torch.LongTensor([new_labels])
        return tokens, new_labels

    def _get_sentence_tokens_and_labels(self, sentence, labels):
        tokens = []
        new_labels = []
        for word, label in zip(sentence, labels):
            tokenized = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            tokens.extend(tokenized)
            new_labels.extend([label] * len(tokenized))
        return tokens, new_labels

    @staticmethod
    def _add_padding(tokens, labels, max_len=512, pad_token=0):
        for idx, token in enumerate(tokens):
            diff = max_len - token.shape[-1]
            if diff < 0:
                token = token[:, :max_len]
                tokens[idx] = token
                label = labels[idx]
                label = label[:, :max_len]
                labels[idx] = label
            else:
                padding = torch.ones((1, diff), dtype=torch.long) * pad_token
                token = torch.cat([token, padding], dim=1)
                label = labels[idx]
                label = torch.cat([label, padding], dim=1)
                tokens[idx] = token
                labels[idx] = label
        return tokens, labels

    @staticmethod
    def _collect_unique_labels(documents):
        labels = []
        for document in documents:
            for segment in document.segments:
                for lbls in segment.sentences_labels:
                    labels.extend(lbls)
        all_labels = list(set(labels))
        int_labels = {}
        for idx, label in enumerate(all_labels):
            int_labels[label] = idx
        return int_labels

    def _transform_labels(self, documents):
        for document in documents:
            for i, sentence_labels in enumerate(document.sentences_labels):
                for j, label in enumerate(sentence_labels):
                    document.sentences_labels[i][j] = self.labels_to_int[label]

    @staticmethod
    def _collect_instances(documents):
        sentences, labels = [], []
        for i, document in enumerate(documents):
            sentences.extend(document.sentences)
            labels.extend(document.sentences_labels)
        return sentences, labels

    def _collect_data(self, documents, other_label="O"):
        all_segments, stance_segments, relations, stances = [], [], {}, {}
        if exists(join(self.resources_folder, self.segments_pickle_file)):
            with open(join(self.resources_folder, self.segments_pickle_file), "rb") as f:
                all_segments, stance_segments, relations, stances = pickle.load(f)
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
                        segment.tokens = self._get_encoded_data(segment)
                        segment.tokens = self._add_relation_padding(segment.tokens)
                        self.app_logger.debug("Encoded data: {}".format(segment.tokens))
                        if "claim" in segment.arg_type:
                            self.app_logger.debug(
                                "Segment is added in the stance list. arg type: {}".format(segment.arg_type))
                            stance_segments.append(segment)
                    all_segments.append(segment)
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
                pickle.dump((all_segments, stance_segments, relations, stances), f)
        return all_segments, stance_segments, relations, stances

    def _get_encoded_data(self, segment):
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

    @staticmethod
    def _add_relation_padding(tokens, max_len=512, pad_token=0):
        diff = max_len - tokens.shape[-1]
        if diff < 0:
            tokens = tokens[:, :max_len]
        else:
            padding = torch.ones((1, diff), dtype=torch.long) * pad_token
            tokens = torch.cat([tokens, padding], dim=1)
        return tokens

    def _collect_relations(self, segments, relations, pickle_file, kind="relation"):
        encoded_labels = (self.relations_to_int, self.int_to_relations) if kind == "relation" else \
            (self.stance_to_int, self.int_to_stance)
        self.app_logger.debug("Collecting relations")
        if exists(join(self.resources_folder, pickle_file)):
            with open(join(self.resources_folder, pickle_file), "rb") as f:
                return pickle.load(f)
        array_size = len(segments)
        data, labels, initial_data = [], [], []
        for i in range(array_size):
            for j in range(array_size):
                if i == j:
                    continue
                arg1 = segments[i]
                arg2 = segments[j]
                label = self.relations_to_int["other"] if kind == "relation" else self.stance_to_int["other"]
                try:
                    relation = relations[(arg1.segment_id, arg2.segment_id)]
                    label = self.relations_to_int[relation] if kind == "relation" else \
                        self.stance_to_int[relation]
                except(KeyError, Exception):
                    pass
                self.app_logger.debug("Averaging segments: {} and {}".format(arg1.text, arg2.text))
                self.app_logger.debug("Segments relation: {}".format(label))
                input_data = (arg1.tokens + arg2.tokens) / 2
                initial_data.append((arg1, arg2, label))
                data.append(input_data)
                labels.append(label)
        with open(join(self.resources_folder, pickle_file), "wb") as f:
            pickle.dump((data, labels, initial_data, encoded_labels), f)
        return data, labels, initial_data, encoded_labels
