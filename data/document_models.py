import re
from typing import List

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
