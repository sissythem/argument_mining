import itertools
import json
import pickle
from os.path import join

import torch
from ellogon import tokeniser
from transformers import BertTokenizer


class ArgumentMining:

    def __init__(self, app_config):
        self.app_config = app_config
        self.resources_folder = app_config.resources_path
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.adu_classifiers = []
        self.relation_classifiers = []
        self.stance_classifiers = []
        self.best_adu_classifier = None
        self.best_relation_classifier = None
        self.best_stance_classifier = None
        self.device_name = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.max_len = self.properties["preprocessing"]["max_len"]
        self.pad_token = self.properties["preprocessing"]["pad_token"]
        self.tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        with open(join(self.resources_folder, "adu_labels.pkl"), "rb") as f:
            self.adu_lbls_to_int, self.int_to_adu_lbls = pickle.load(f)
        self.relations_to_int = {"other": 0, "support": 1, "attack": 2}
        self.int_to_relations = {0: "other", "support": 1, "attack": 2}
        self.stance_to_int = {"other": 0, "for": 1, "against": 2}
        self.int_to_stance = {0: "other", 1: "for", 2: "against"}

    def load(self, adu_model_file_path, relations_model_file_path, stance_model_file_path):
        with open(join(self.app_config.output_path, adu_model_file_path), "rb") as f:
            print("Loading outputs from", f.name)
            self.adu_classifiers = pickle.load(f)
        with open(join(self.app_config.output_path, relations_model_file_path), "rb") as f:
            print("Loading outputs from", f.name)
            self.relation_classifiers = pickle.load(f)
        with open(join(self.app_config.output_path, stance_model_file_path), "rb") as f:
            print("Loading outputs from", f.name)
            self.stance_classifiers = pickle.load(f)
        self.best_adu_classifier = self.select_best_classifier(self.adu_classifiers).to(self.device_name)
        self.best_relation_classifier = self.select_best_classifier(self.relation_classifiers).to(self.device_name)
        self.best_stance_classifier = self.select_best_classifier(self.stance_classifiers).to(self.device_name)

    def predict(self, documents):
        self.app_logger.info("Doing token classification")
        for d, document in enumerate(documents):
            sentences = tokeniser.tokenise_no_punc(document.content)
            all_tokens = []
            all_predictions = []
            for s, sentence in enumerate(sentences):
                if len(sentence) > self.max_len:
                    print(f"(!!!) Sentence length #{s}:  {len(sentence)} but max len is {self.max_len}")
                tokens = self.tokenizer(sentence, is_split_into_words=True, add_special_tokens=True,
                                        padding="max_length", truncation=True, max_length=self.max_len)["input_ids"]
                tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device_name)
                predictions = self.best_adu_classifier.forward(tokens)
                all_tokens.append(tokens)
                all_predictions.append(predictions)
            segments, adus, new_tokens = self._get_segments(sentences=sentences, predictions=all_predictions,
                                                            tokens=all_tokens)

            relations, stances = [], []
            for stup in itertools.combinations(new_tokens, 2):
                idx1 = new_tokens.index(stup[0])
                idx2 = new_tokens.index(stup[1])
                segment1 = segments[idx1]
                segment2 = segments[idx2]
                relation_preds = self.best_relation_classifier.forward(stup)
                stance_preds = self.best_stance_classifier.forward(stup)
                relations.append(((segment1, segment2), relation_preds))
                stances.append(((segment1, segment2), stance_preds))
            relations = self._get_relations(preds=relations)
            stances = self._get_relations(preds=stances, kind="stance")
            doc = {
                "id": document.document_id,
                "link": "",
                "description": "",
                "date": "",
                "tags": [],
                "document_link": "",
                "publishedAt": "",
                "crawledAt": "",
                "domain": "",
                "netloc": "",
                "content": document.content,
                "annotations": {
                    "ADUs": [],
                    "Relations": [],
                    "Stance": []
                }
            }
            counter = 1
            for idx, segment in enumerate(segments):
                seg = {
                    "id": "T{}".format(counter),
                    "type": adus[idx],
                    "starts": str(document.content.index(segment)),
                    "ends": str(document.content.index(segment) + len(segment)),
                    "segment": segment
                }
                doc["ADUs"].append(seg)
            counter = 1
            segments = doc["ADUs"]
            for relation in relations:
                arg1 = relation[0]
                arg2 = relation[1]
                rel = relation[2]
                arg1_id, arg2_id = self._find_args_in_relation(segments, arg1, arg2)
                rel_dict = {
                    "id": "R{}".format(counter),
                    "type": rel,
                    "arg1": arg1_id,
                    "arg2": arg2_id
                }
                doc["Relations"].append(rel_dict)
            counter = 1
            for s in stances:
                arg1 = s[0]
                arg2 = s[1]
                rel = s[2]
                arg1_id, arg2_id = self._find_args_in_relation(segments, arg1, arg2)
                stance_dict = {
                    "id": "A{}".format(counter),
                    "type": rel,
                    "arg1": arg1_id,
                    "arg2": arg2_id
                }
                doc["Stance"].append(stance_dict)
            with open(self.app_config.out_file_path, "w") as f:
                f.write(json.dumps(doc, indent=4, sort_keys=False))

    def _get_segments(self, sentences, tokens, predictions):
        sentences, tokens = self._unify_sentences_list(sentences, tokens)
        segments = []
        adus = []
        segment_tokens = []
        lbls = sorted(list(self.int_to_adu_lbls.keys()))
        lbls_txt = [self.int_to_adu_lbls[lbl] for lbl in lbls]
        start_lbls = [lbl for lbl in lbls_txt if lbl.startswith("B")]
        idx = 0
        while idx < len(predictions):
            pred = predictions[idx]
            predicted_label = self.int_to_adu_lbls[pred]
            if predicted_label in start_lbls:
                toks = []
                adu_label = predicted_label.replace("B-", "")
                adus.append(adu_label)
                segment_text = sentences[idx]
                toks.append(tokens[idx])
                next_correct = "I-{}".format(adu_label)
                lbl = next_correct
                while lbl == next_correct:
                    idx += 1
                    if idx >= len(predictions):
                        break
                    next_pred = predictions[idx]
                    lbl = self.int_to_adu_lbls[next_pred]
                    if lbl != next_correct:
                        break
                    segment_text += sentences[idx]
                segments.append(segment_text)
                segment_tokens.append(toks)
                idx += 1
        return segments, adus, segment_tokens

    def _get_relations(self, preds, kind="relations"):
        relations = []
        other_int = self.relations_to_int["other"] if kind == "relations" else self.stance_to_int["other"]
        for pred in preds:
            args = pred[0]
            pred = pred[1]
            if pred == other_int:
                continue
            arg1 = args[0]
            arg2 = args[1]
            relation = self.int_to_relations[pred] if kind == "relations" else self.int_to_stance[pred]
            relations.append((arg1, arg2, relation))
        return relations

    @staticmethod
    def _unify_sentences_list(sentences, all_tokens):
        new_sentences = []
        new_tokens = []
        for sentence in sentences:
            for word in sentence:
                new_sentences.append(word)
        for tokens in all_tokens:
            for t in tokens:
                new_tokens.append(t)
        return new_sentences, new_tokens

    @staticmethod
    def select_best_classifier(classifiers):
        # sort by mean test acc.
        c = sorted(classifiers, key=lambda x: x.test_accuracy)
        return c[-1]

    @staticmethod
    def _find_args_in_relation(segments, arg1, arg2):
        arg1_id, arg2_id = "", ""
        for segment in segments:
            if segment["segment"] == arg1:
                arg1_id = segment["id"]
            elif segment["segment"] == arg2:
                arg2_id = segment["id"]
        if not arg1_id:
            arg1_id = arg1
        if not arg2_id:
            arg2_id = arg2
        return arg1_id, arg2_id
