import hashlib
import json
import pickle
import random
from os import listdir
from os.path import join, exists
from typing import List, Dict, AnyStr, Union, Tuple
from flair.data import Sentence
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from src.utils import utils
from src.utils.config import AppConfig


class DataUpSampler:

    def __init__(self, app_config: AppConfig):
        self.app_config: AppConfig = app_config
        self.app_logger = app_config.app_logger
        self.data_folder = app_config.dataset_folder

    def oversample(self, task_kind: AnyStr, file_kind: AnyStr, total_num: Union[Dict, int]):
        """
        Oversampling of imbalanced datasets. For now it is implemented only for relations/stance datasets. The new
        datasets are exported into a csv file.

        Args
            task_kind (str): can take values from --> adu, rel or stance
            file_kind (str): possible values are --> train, test, dev
            total_num(int or dict): the total number to augment the minority classes
        """
        data_path = join(self.data_folder, task_kind)
        filename = f"{file_kind}.csv"
        file_path = join(data_path, filename)
        df = pd.read_csv(file_path, sep="\t", index_col=None, header=None)
        if task_kind == "adu":
            if not type(total_num) == dict:
                self.app_logger.warning(
                    "For ADU oversampling total_num should be a dict with the labels & their final count")
                self.app_logger.warning("Labels that have equal or greater number of instance will be kept the same")
            df = self.oversample_adus(data=df, desired_lbl_count=total_num)
        else:
            df = self.oversample_relations(df=df, rel=task_kind, total_num=total_num)
        filename = filename.replace(".csv", "")
        new_file = f"{filename}_oversample.csv"
        output_filepath = join(data_path, new_file)
        if not exists(output_filepath):
            df.to_csv(output_filepath, sep='\t', index=False, header=True)

    @staticmethod
    def oversample_adus(data: pd.DataFrame, desired_lbl_count: Dict):
        df_list = np.split(data, data[data.isnull().all(1)].index)
        df_list = [df.dropna() for df in df_list]
        labels_dict = {"B-major_claim": [], "I-major_claim": [], "B-claim": [], "I-claim": [], "B-premise": [],
                       "I-premise": []}
        for df in df_list:
            labels = list(df[1])
            for lbl in labels_dict.keys():
                if lbl in labels:
                    labels_dict[lbl].append(df)
        for lbl, desired_count in desired_lbl_count.items():
            dfs = labels_dict[lbl]
            append_idxs = []
            num_instances = len(dfs)
            ixs = range(num_instances)
            while True:
                if desired_count <= num_instances + len(append_idxs):
                    break
                num_append = min(desired_count - num_instances + len(append_idxs), num_instances)
                append_idxs.extend(random.sample(ixs, num_append))
            dfs.extend([dfs[i] for i in append_idxs])
        new_df_list = []
        for _, df_list in labels_dict.items():
            for df in df_list:
                empty_row = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]])
                new_df_list.append(df)
                new_df_list.append(empty_row)
        return pd.concat(new_df_list, axis=0)

    @staticmethod
    def oversample_relations(df: pd.DataFrame, rel: AnyStr, total_num: int):
        texts = list(df[0])
        indices = [texts.index(x) for x in texts]
        labels = list(df[1])
        numeric_labels = []
        unique_labels = set(labels)
        count = 0
        lbl_dict = {}
        str_to_num = {}
        # map labels to integers
        for lbl in unique_labels:
            lbl_dict[count] = lbl
            str_to_num[lbl] = count
            count += 1
        # get int labels
        for lbl in labels:
            numeric_labels.append(str_to_num[lbl])
        data = np.asarray(indices).reshape(-1, 1)
        labels = np.asarray(numeric_labels).reshape(-1, 1)
        if rel == "rel":
            num_support = str_to_num["support"]
            num_attack = str_to_num["attack"]
            sampling_strategy = {num_support: total_num, num_attack: total_num}
        else:
            num_against = str_to_num["against"]
            sampling_strategy = {num_against: total_num}
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy)
        data, labels = sampler.fit_resample(data.reshape(-1, 1), labels)
        data = data.squeeze()
        labels = labels.squeeze()
        data = list(data)
        labels = list(labels)
        labels = [lbl_dict[lbl] for lbl in labels]
        data = [texts[x] for x in data]
        for idx, text in enumerate(data):
            text = text.replace("\t", " ")
            text = utils.replace_multiple_spaces_with_single_space(text)
            data[idx] = text
        new_df = pd.DataFrame(columns=["text", "label"])
        new_df["text"] = data
        new_df["label"] = labels
        return new_df


class CsvCreator:

    def __init__(self, app_config: AppConfig, pickle_file="documents.pkl"):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.pickle_file = pickle_file
        self.oversampling_prop = app_config.properties["prep"]["oversampling"]
        if self.oversampling_prop is not None and self.oversampling_prop and type(self.oversampling_prop) == dict:
            self.upsampling = DataUpSampler(app_config=app_config)

    def load_adus(self, folder="kasteli"):
        self.app_logger.debug("Running ADU preprocessing")
        out_file_path = join(self.app_config.dataset_folder, "adu", "train.csv")
        if not exists(out_file_path):
            documents_path = join(self.app_config.dataset_folder, "initial", folder, self.pickle_file)
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
                        doc_sentence_counter += 1
                    df.loc[row_counter] = ["", "", "", "", "", ""]
                    sentence_counter += 1
                    row_counter += 1
            self.app_logger.debug("Finished building dataframe. Saving...")
            df.to_csv(out_file_path, sep='\t', index=False, header=False)
            self.app_logger.debug("Dataframe saved!")
        if self.oversampling_prop:
            adu_config = self.oversampling_prop.get("adu", None)
            if adu_config:
                self.upsampling.oversample(task_kind="adu", file_kind="train", total_num=adu_config)

    def load_similarities(self, do_oversample=False):
        self.app_logger.debug("Reading UKP aspect corpus")
        data_path = join(self.app_config.dataset_folder, "sim")
        data_csv = "UKP_ASPECT.tsv"
        data_file_path = join(data_path, data_csv)
        output_filepath = join(data_path, "train.csv")
        if not exists(output_filepath):
            df = pd.read_csv(data_file_path, sep="\t", header=0, index_col=None)
            new_df = pd.DataFrame(columns=["sentence", "label", "topic"])
            row_counter = 0
            for index, row in df.iterrows():
                topic, sentence1, sentence2, label = row
                final_text = f"[CLS] {sentence1} [SEP] {sentence2}"
                new_df.loc[row_counter] = [final_text, label, topic]
                row_counter += 1

            new_df.to_csv(output_filepath, sep='\t', index=False, header=False)
            self.app_logger.debug("Dataframe saved!")
        if do_oversample:
            # TODO if needed
            pass

    def load_relations_and_stance(self, folder="kasteli"):
        relations, stances = self._get_relations(folder=folder)
        self._save_rel_df(rel_list=relations, folder="rel", filename="train.csv")
        self._save_rel_df(rel_list=stances, folder="stance", filename="train.csv")
        if self.oversampling_prop:
            rel_num = self.oversampling_prop.get("rel", None)
            stance_num = self.oversampling_prop.get("stance", None)
            if rel_num is not None and type(rel_num) == int:
                self.upsampling.oversample(task_kind="rel", file_kind="train", total_num=rel_num)
            if stance_num is not None and type(stance_num) == int:
                self.upsampling.oversample(task_kind="stance", file_kind="train", total_num=stance_num)

    def _get_relations(self, folder):
        documents_path = join(self.app_config.dataset_folder, "initial", folder, self.pickle_file)
        self.app_logger.debug("Loading documents from pickle file")
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
        self.app_logger.debug("Documents are loaded")
        relations, stances = [], []
        for document in documents:
            self.app_logger.debug(f"Processing relations for document: {document.document_id}")
            major_claims, claims, premises, relation_pairs, stance_pairs = self._collect_segments(document)
            relations += utils.collect_relation_pairs(parents=major_claims, children=claims,
                                                      relation_pairs=relation_pairs)
            relations += utils.collect_relation_pairs(parents=claims, children=premises,
                                                      relation_pairs=relation_pairs)
            self.app_logger.debug(f"Found {len(relations)} relations")
            stances += utils.collect_relation_pairs(parents=major_claims, children=claims,
                                                    relation_pairs=stance_pairs)
            self.app_logger.debug(f"Found {len(stances)} stance")
        return relations, stances

    def _save_rel_df(self, rel_list, folder, filename):
        data_path = join(self.app_config.dataset_folder, folder)
        output_filepath = join(data_path, filename)
        if exists(output_filepath):
            return
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


class Relation:
    def __init__(self, relation_id, document_id, arg1, arg2, kind, relation_type):
        self.relation_id: str = relation_id
        self.document_id: int = document_id
        self.arg1: Segment = arg1
        self.arg2: Segment = arg2
        self.kind: str = kind
        self.relation_type: str = relation_type


class Document:

    def __init__(self, logger, document_id, name, content, annotations):
        self.app_logger = logger
        self.document_id: int = document_id
        self.name: str = name
        self.content: str = content
        self.annotations: List[dict] = annotations
        self.segments: List[Segment] = []
        self.relations: List[Relation] = []
        self.stance: List[Relation] = []
        self.sentences: List = []
        self.sentences_labels: List = []

    def update_segments(self, other_label="O"):
        sentences = utils.join_sentences(tokenized_sentences=self.sentences)
        for idx, sentence in enumerate(sentences):
            segments = self._get_segments(sentence=sentence)
            sentence_split = self.sentences[idx]
            if segments:
                tokens_lbls = self._get_tokens_and_labels(sentence_tokens=sentence_split, segments=segments)
                sentence_split, sentence_labels = utils.bio_tag_lbl_per_token(tokens_labels_tuple=tokens_lbls)
            else:
                label = other_label
                sentence_split = tuple([token for token in sentence_split if token is not None or token != ""])
                sentence_labels = tuple([label for _ in sentence_split])

            self.sentences[idx] = sentence_split
            self.sentences_labels.append(sentence_labels)

    def _get_segments(self, sentence: AnyStr) -> List[Segment]:
        segments = []
        for segment in self.segments:
            text = segment.text
            if text in sentence or sentence in text:
                segments.append(segment)
        return segments

    @staticmethod
    def _get_tokens_and_labels(sentence_tokens: Union[Tuple[AnyStr], List[AnyStr]],
                               segments: List[Segment],
                               other_label: AnyStr = "O",
                               repl_char: AnyStr = "="):
        """
        Creates a list for each sentence token, containing the token and the respective label, without BIO tagging

        Args
            | sentence_tokens (list): a list of the tokens of the sentence
            | segment (Segment): the respective Segment obj
            | other_label (str): the label of non argument tokens

        Returns
            | list: a list of tuples, each containing a token and a label
        """
        tokens, labels = [], []
        sentence_tokens = list(sentence_tokens)
        # collect tokens & labels for all the found segments
        for segment in segments:
            for segment_sentence in segment.sentences:
                for segment_token in segment_sentence:
                    tokens.append(segment_token)
                    labels.append(segment.arg_type)
        counter = 0
        overlapping = []
        # get overlapping tokens between segments and sentence. Replace the tokens with the repl_char
        for idx, token in enumerate(tokens):
            label = labels[idx]
            while counter < len(sentence_tokens):
                sentence_token = sentence_tokens[counter]
                if sentence_token == token:
                    overlapping.append((sentence_token, label))
                    sentence_tokens[counter] = repl_char * 10
                    counter += 1
                    break
                counter += 1
        tkn_idx = 0
        # collect all the tokens. Those not replaced will have the other label
        segment_tokens = []
        for sentence_token in sentence_tokens:
            if sentence_token == repl_char * 10:
                sgmnt_tuple = overlapping[tkn_idx]
                segment_tokens.append(sgmnt_tuple)
                tkn_idx += 1
            else:
                segment_tokens.append((sentence_token, other_label))
        return segment_tokens


class ClarinLoader:

    def __init__(self, app_config: AppConfig, folder="kasteli", json_file="kasteli.json", pickle_file="documents.pkl"):
        self.app_config: AppConfig = app_config
        self.app_logger = self.app_config.app_logger
        self.dataset_path = join(self.app_config.dataset_folder, "initial", folder)
        self.json_file = json_file
        self.pickle_file = pickle_file

    def load(self) -> List[Document]:
        """
        Loads the Clarin dataset either from a pickle file (existing processing) or from the json

        Returns
            | list: the loaded documents
        """
        path_to_pickle = join(self.dataset_path, self.pickle_file)
        if exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                return pickle.load(f)
        path_to_data = join(self.dataset_path, self.json_file)
        with open(path_to_data, "r") as f:
            content = json.loads(f.read())
        documents = content["data"]["documents"]
        docs = []
        for doc in documents:
            document = self.create_document(doc=doc)
            docs.append(document)
        with open(join(self.dataset_path, self.pickle_file), "wb") as f:
            pickle.dump(docs, f)
        return docs

    def create_document(self, doc: Dict):
        """
        Process a document dict into a Document object

        Args
            | doc (dict): the document data

        Returns
            | Document: a Document object
        """
        document = Document(logger=self.app_logger, document_id=doc["id"], name=doc["name"], content=doc["text"],
                            annotations=doc["annotations"])
        document.sentences = utils.tokenize(text=document.content)
        self.app_logger.debug(f"Processing document with id {document.document_id}")
        for annotation in document.annotations:
            annotation_id = annotation["_id"]
            spans = annotation["spans"]
            segment_type = annotation["type"]
            attributes = annotation["attributes"]
            if utils.is_old_annotation(attributes):
                continue
            if segment_type == "argument":
                span = spans[0]
                segment = self.create_segment(span=span, document_id=document.document_id,
                                              annotation_id=annotation_id, attributes=attributes)
                document.segments.append(segment)
            elif segment_type == "argument_relation":
                relation = self.create_relation(segments=document.segments, attributes=attributes,
                                                annotation_id=annotation_id, document_id=document.document_id)
                if relation:
                    if relation.kind == "relation":
                        document.relations.append(relation)
                    else:
                        document.stance.append(relation)
        document.segments.sort(key=lambda x: x.char_start)
        document.update_segments()
        document.segments.sort(key=lambda x: x.char_start)
        return document

    @staticmethod
    def create_segment(span, document_id, annotation_id, attributes):
        segment_text = span["segment"]
        segment = Segment(segment_id=annotation_id, document_id=document_id, text=segment_text,
                          char_start=span["start"], char_end=span["end"], arg_type=attributes[0]["value"])
        segment.sentences = utils.tokenize(text=segment.text)
        segment.sentences, segment.sentences_labels = utils.bio_tagging(sentences=segment.sentences,
                                                                        label=segment.arg_type)
        return segment

    @staticmethod
    def create_relation(segments, attributes, annotation_id, document_id):
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
        if arg1 is None or arg2 is None:
            return None
        return Relation(relation_id=annotation_id, document_id=document_id, arg1=arg1,
                        arg2=arg2, kind=kind, relation_type=relation_type)


class EssayLoader:

    def __init__(self, app_config: AppConfig, pickle_file="documents.pkl"):
        self.app_config: AppConfig = app_config
        self.app_logger = self.app_config.app_logger
        self.dataset_path = join(self.app_config.dataset_folder, "initial", "essays")
        self.eng_data_path = join(self.dataset_path, "english")
        self.gr_data_path = join(self.dataset_path, "greek")
        self.pickle_file = pickle_file

    def load(self):
        path_to_pickle = join(self.dataset_path, self.pickle_file)
        if exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                return pickle.load(f)
        eng_data = self.get_json(path=self.eng_data_path)
        gr_data = self.get_json(path=self.gr_data_path)
        documents = {**eng_data, **gr_data}
        docs = []
        for key, doc in documents.items():
            document = self.create_document(doc=doc, name=key)
            docs.append(document)
        with open(join(self.dataset_path, self.pickle_file), "wb") as f:
            pickle.dump(docs, f)
        return docs

    def create_document(self, doc: Dict, name: AnyStr):
        """
        Process a document dict into a Document object

        Args
            | doc (dict): the document data

        Returns
            | Document: a Document object
        """
        document = Document(logger=self.app_logger, document_id=doc["id"], name=name, content=doc["content"],
                            annotations=doc["annotations"])
        document.sentences = utils.tokenize(text=document.content)
        self.app_logger.debug(f"Processing document with id {document.document_id}")
        adus = document.annotations["ADUs"]
        relations = document.annotations["Relations"]
        major_claim_id = -1
        for adu in adus:
            segment_id = adu["id"]
            segment_type = adu["type"]
            if segment_type == "MajorClaim":
                major_claim_id = segment_id
            if segment_type == "MajorClaim":
                segment_type = "major_claim"
            elif segment_type == "Claim":
                segment_type = "claim"
            else:
                segment_type = "premise"
            starts = adu["starts"]
            ends = adu["ends"]
            segment = adu["segment"]
            segment = self.create_segment(document_id=doc["id"], segment_id=segment_id, segment_type=segment_type,
                                          starts=starts, ends=ends, text=segment)
            if segment:
                document.segments.append(segment)
        for relation in relations:
            if not relation:
                continue
            rel_type = relation["type"]
            rel_type = "support" if rel_type == "supports" else "attack"
            rel = self.create_relation(segments=document.segments, relation_id=relation["id"], rel_type=rel_type,
                                       kind="relation", document_id=document.document_id, arg1_id=relation["arg1"],
                                       arg2_id=relation["arg2"])
            if rel:
                document.relations.append(rel)
        rel_counter = int(document.relations[-1].relation_id.replace("R", ""))
        for adu in adus:
            if "stance" not in adu.keys() or not adu["stance"]:
                continue
            stance = adu["stance"][0]
            stance_type = "for" if stance["type"] == "For" else "against"
            rel_type = "support" if stance_type == "for" else "attack"
            st = self.create_relation(segments=document.segments, relation_id=stance["id"], rel_type=stance_type,
                                      kind="stance", document_id=document.document_id, arg1_id=adu["id"],
                                      arg2_id=major_claim_id)
            rel_counter += 1
            rel = self.create_relation(segments=document.segments, relation_id=rel_counter, rel_type=rel_type,
                                       kind="relation", document_id=document.document_id, arg1_id=adu["id"],
                                       arg2_id=major_claim_id)
            if st:
                document.stance.append(st)
            if rel:
                document.relations.append(rel)
        document.segments.sort(key=lambda x: x.char_start)
        document.update_segments()
        document.segments.sort(key=lambda x: x.char_start)
        return document

    @staticmethod
    def create_segment(document_id, segment_id, segment_type, starts, ends, text):
        if not text or text == "nan" or type(text) == float:
            return None
        segment = Segment(segment_id=segment_id, document_id=document_id, text=text,
                          char_start=starts, char_end=ends, arg_type=segment_type)
        segment.sentences = utils.tokenize(text=segment.text)
        segment.sentences, segment.sentences_labels = utils.bio_tagging(sentences=segment.sentences,
                                                                        label=segment.arg_type)
        return segment

    @staticmethod
    def create_relation(segments, relation_id, rel_type, kind, document_id, arg1_id, arg2_id):
        arg1, arg2 = None, None
        for seg in segments:
            if seg.segment_id == arg1_id:
                arg1 = seg
            elif seg.segment_id == arg2_id:
                arg2 = seg
        if arg1 is None or arg2 is None:
            return None
        return Relation(relation_id=relation_id, document_id=document_id, arg1=arg1, arg2=arg2, kind=kind,
                        relation_type=rel_type)

    def get_json(self, path):
        files = listdir(path)
        processed_files = {}
        if files:
            for file in files:
                if file.endswith(".txt") or file.endswith(".ann"):
                    filename = file.split(".")[0]
                    # if filename does not already exists in the processed_files dict
                    # add fields
                    if filename not in processed_files.keys():
                        hash_id = hashlib.md5(filename.encode())
                        processed_files[filename] = {"id": hash_id.hexdigest(), "link": "", "description": "",
                                                     "date": "",
                                                     "tags": [], "publishedAt": "", "crawledAt": "", "domain": "",
                                                     "netloc": ""}
                    # add content and title
                    if file.endswith(".txt"):
                        with open(join(path, file), "r") as f:
                            lines = f.readlines()
                            lines = [line for line in lines if line and line != "\n"]
                        processed_files[filename]["content"] = " ".join(lines)
                    elif file.endswith(".ann"):
                        df = pd.read_csv(join(path, file), index_col=None, header=None, sep="\t")
                        data = self.process_df(df)
                        processed_files[filename]["annotations"] = data
                        processed_files[filename]["annotations"]["document_link"] = ""
        return processed_files

    @staticmethod
    def process_df(df):
        data = {"ADUs": [], "Relations": []}
        for index, row in df.iterrows():
            if row[0].startswith("T"):
                adu_type = row[1].split(" ")
                adu = {"id": row[0], "type": adu_type[0], "starts": adu_type[1], "ends": adu_type[2], "segment": row[2]}
                data["ADUs"].append(adu)
            elif row[0].startswith("R"):
                rel_type = row[1].split(" ")
                data["Relations"].append({"id": row[0], "type": rel_type[0], "arg1": rel_type[1].split(":")[1],
                                          "arg2": rel_type[2].split(":")[1]})
            else:  # stance
                rel = row[1].split(" ")
                if data["ADUs"]:
                    for adu in data["ADUs"]:
                        if adu["id"] == rel[1]:
                            if "stance" not in adu.keys():
                                adu["stance"] = []
                            adu["stance"].append({"id": row[0], "type": rel[2]})
        return data


class CustomSentence(Sentence):

    def __init__(self, sentence_tuple):
        tokens_tuple, start_idx, end_idx = sentence_tuple
        self.start_idx = start_idx
        self.end_idx = end_idx
        only_tokens = [token[0] for token in tokens_tuple]
        self.text = utils.join_sentence(sentence=only_tokens)
        self.tokens = []
        for token in tokens_tuple:
            tok = CustomToken(text=token[0], start=token[1], end=token[2])
            self.tokens.append(tok)
        super(CustomSentence, self).__init__(self.text)


class CustomToken:

    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
