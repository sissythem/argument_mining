from typing import List
from utils.utils import Utilities
from utils.config import AppConfig


class ClarinLoader:

    def __init__(self, utilities: Utilities):
        self.utilities = utilities
        self.app_config: AppConfig = utilities.app_config
        self.app_logger = self.app_config.app_logger

    def create_document(self, doc: dict):
        document = Document(utilities=self.utilities, document_id=doc["id"], name=doc["name"], content=doc["text"],
                            annotations=doc["annotations"])
        document.sentences = self.utilities.tokenize(text=document.content)
        self.app_logger.debug(f"Processing document with id {document.document_id}")
        for annotation in document.annotations:
            annotation_id = annotation["_id"]
            spans = annotation["spans"]
            segment_type = annotation["type"]
            attributes = annotation["attributes"]
            if self.utilities.is_old_annotation(attributes):
                continue
            if segment_type == "argument":
                span = spans[0]
                segment = self.create_segment(span=span, document_id=document.document_id,
                                              annotation_id=annotation_id, attributes=attributes)
                document.segments.append(segment)
            elif segment_type == "argument_relation":
                relation = self.create_relation(segments=document.segments, attributes=attributes,
                                                annotation_id=annotation_id, document_id=document.document_id)
                if relation.kind == "relation":
                    document.relations.append(relation)
                else:
                    document.stance.append(relation)
        document.segments.sort(key=lambda x: x.char_start)
        document.update_segments(repl_char=self.app_config.properties["preprocessing"]["repl_char"],
                                 other_label=self.app_config.properties["preprocessing"]["other_label"])
        document.update()
        document.segments.sort(key=lambda x: x.char_start)
        return document

    def create_segment(self, span, document_id, annotation_id, attributes):
        pass

    def create_relation(self, segments, attributes, annotation_id, document_id):
        pass


class Document:

    def __init__(self, utilities: Utilities, document_id, name, content, annotations):
        self.utilities: Utilities = utilities
        self.app_config: AppConfig = utilities.app_config
        self.app_logger = self.app_config.app_logger
        self.document_id: int = document_id
        self.name: str = name
        self.content: str = content
        self.annotations: List[dict] = annotations
        self.segments: List[Segment] = []
        self.relations: List[Relation] = []
        self.stance: List[Relation] = []
        self.sentences: List = []
        self.sentences_labels: List = []

    def update(self):
        pass

    def update_segments(self):
        pass


class Segment:
    def __init__(self, segment_id, document_id, text, char_start, char_end, arg_type):
        self.segment_id: str = segment_id
        self.document_id: int = document_id
        self.text: str = text
        self.char_start: int = char_start
        self.char_end: int = char_end
        self.arg_type: str = arg_type
        self.sentences: List = []
        self.sentences_labels: List = []

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
