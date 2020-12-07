import json
import pickle
from os.path import join, exists

from ellogon import tokeniser

from data.document_models import Document, Segment, Relation


class DataLoader:

    def __init__(self, app_config):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.resources_folder = app_config.resources_path
        self.data_file = app_config.data_file
        self.pickle_file = app_config.documents_pickle

    def load(self):
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
