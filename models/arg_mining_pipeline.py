import json
import pickle
from os.path import join

import torch
from ellogon import tokeniser
from transformers import BertTokenizer

from base import utils


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
        self.device_name = app_config.device_name
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
            self.app_logger.debug("Loading ADU classifiers from {}".format(f.name))
            self.adu_classifiers = pickle.load(f)
        with open(join(self.app_config.output_path, relations_model_file_path), "rb") as f:
            self.app_logger.debug("Loading relation classifiers from {}".format(f.name))
            self.relation_classifiers = pickle.load(f)
        with open(join(self.app_config.output_path, stance_model_file_path), "rb") as f:
            self.app_logger.debug("Loading stance classifiers from {}".format(f.name))
            self.stance_classifiers = pickle.load(f)
        self.best_adu_classifier = self.select_best_classifier(self.adu_classifiers).to(self.device_name)
        self.best_relation_classifier = self.select_best_classifier(self.relation_classifiers).to(self.device_name)
        self.best_stance_classifier = self.select_best_classifier(self.stance_classifiers).to(self.device_name)

    def predict(self, documents):
        self.app_logger.info("Doing token classification")
        for d, document in enumerate(documents):
            doc = self._get_initial_doc(document)
            sentences = tokeniser.tokenise_no_punc(document.content)
            all_tokens = []
            all_predictions = []
            for s, sentence in enumerate(sentences):
                if len(sentence) > self.max_len:
                    self.app_logger.warning(f"(!!!) Sentence length #{s}:{len(sentence)} but max len is {self.max_len}")
                tokens = self.tokenizer(sentence, is_split_into_words=True, add_special_tokens=True,
                                        padding="max_length", truncation=True, max_length=self.max_len)["input_ids"]
                tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device_name)
                predictions = self.best_adu_classifier.forward(tokens)
                all_tokens.append(tokens)
                all_predictions.append(predictions)
            segments, adus, new_tokens = self._get_segments(sentences=sentences, predictions=all_predictions,
                                                            all_tokens=all_tokens)
            major_claims, major_claims_tokens, claims, claims_tokens, premises, premises_tokens = self._split_segments(
                segments=segments, adus=adus, segment_tokens=new_tokens)
            if new_tokens:
                max_len = self.app_config.properties["preprocessing"]["max_len"]
                pad_token = self.app_config.properties["preprocessing"]["pad_token"]
                cls_id = self.tokenizer.cls_token_id
                sep_id = self.tokenizer.sep_token_id
                relations, stances = [], []
                for i in range(len(major_claims)):
                    for j in range(len(claims)):
                        arg1, arg1_txt, arg2, arg2_txt = self._get_relation_args(idx1=i, idx2=j,
                                                                                 token1=major_claims_tokens,
                                                                                 tokens2=claims_tokens,
                                                                                 text1=major_claims,
                                                                                 text2=claims, cls_id=cls_id,
                                                                                 sep_id=sep_id, max_len=max_len,
                                                                                 pad_token=pad_token)
                        relation_preds = self.best_relation_classifier.forward((arg1, arg2))
                        stance_preds = self.best_stance_classifier.forward((arg1, arg2))
                        relations.append(((arg1_txt, arg2_txt), relation_preds))
                        stances.append(((arg1_txt, arg2_txt), stance_preds))
                for i in range(len(claims)):
                    for j in range(len(premises)):
                        arg1, arg1_txt, arg2, arg2_txt = self._get_relation_args(idx1=i, idx2=j,
                                                                                 token1=claims_tokens,
                                                                                 tokens2=premises_tokens,
                                                                                 text1=claims,
                                                                                 text2=premises, cls_id=cls_id,
                                                                                 sep_id=sep_id, max_len=max_len,
                                                                                 pad_token=pad_token)
                        relation_preds = self.best_relation_classifier.forward((arg1, arg2))
                        relations.append(((arg1_txt, arg2_txt), relation_preds))
                relations = self._get_relations(preds=relations)
                stances = self._get_relations(preds=stances, kind="stance")
                counter = 1
                for idx, segment in enumerate(segments):
                    seg = {
                        "id": "T{}".format(counter),
                        "type": adus[idx],
                        "starts": str(document.content.index(segment)),
                        "ends": str(document.content.index(segment) + len(segment)),
                        "segment": segment
                    }
                    doc["annotations"]["ADUs"].append(seg)
                if relations:
                    counter = 1
                    segments = doc["annotations"]["ADUs"]
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
                        doc["annotations"]["Relations"].append(rel_dict)
                if stances:
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
                        doc["annotations"]["Stance"].append(stance_dict)
            filename = "document{}.json".format(document.document_id)
            with open(join(self.app_config.out_files_path, filename), "w") as f:
                f.write(json.dumps(doc, indent=4, sort_keys=False))

    @staticmethod
    def _get_initial_doc(document):
        return {
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

    def _get_segments(self, sentences, all_tokens, predictions):
        self.app_logger.debug("Getting ADUs from predicted sequences")
        segments = []
        adus = []
        segment_tokens = []
        lbls = sorted(list(self.int_to_adu_lbls.keys()))
        lbls_txt = [self.int_to_adu_lbls[lbl] for lbl in lbls]
        start_lbls = [lbl for lbl in lbls_txt if lbl.startswith("B")]
        for i, prediction in enumerate(predictions):
            idx = 0
            prediction = prediction[0]
            sentence = sentences[i]
            tokens = all_tokens[i]
            self.app_logger.debug("Checking sentence: {}".format(sentence))
            self.app_logger.debug("Predictions are: {}".format(prediction))
            try:
                tokens = tokens.to("cpu")
            except(BaseException, Exception):
                pass
            tokens = tokens.numpy()
            tokens = list(tokens.reshape((tokens.shape[1],)))
            tokens, prediction = self._remove_padding(tokens=tokens, predictions=prediction,
                                                      pad_token=self.app_config.properties["preprocessing"][
                                                          "pad_token"])
            while idx < len(prediction):
                pred = prediction[idx]
                predicted_label = self.int_to_adu_lbls[pred]
                if predicted_label in start_lbls:
                    toks = []
                    adu_label = predicted_label.replace("B-", "")
                    self.app_logger.debug("Predicted label: {}".format(adu_label))
                    adus.append(adu_label)
                    segment_text = sentence[idx]
                    toks.append(tokens[idx])
                    next_correct = "I-{}".format(adu_label)
                    lbl = next_correct
                    while lbl == next_correct:
                        idx += 1
                        if idx >= len(prediction):
                            break
                        next_pred = prediction[idx]
                        lbl = self.int_to_adu_lbls[next_pred]
                        if lbl != next_correct:
                            break
                        segment_text += sentence[idx]
                        toks.append(tokens[idx])
                    self.app_logger.debug("Final segment text: {}".format(segment_text))
                    segments.append(segment_text)
                    segment_tokens.append(toks)
                idx += 1
        return segments, adus, segment_tokens

    @staticmethod
    def _split_segments(segments, adus, segment_tokens):
        major_claims, major_claims_tokens = [], []
        claims, claims_tokens = [], []
        premises, premises_tokens = [], []
        for i, adu in adus:
            if "major" in adu:
                major_claims.append(segments[i])
                major_claims_tokens.append(segment_tokens[i])
            elif adu == "claim":
                claims.append(segments[i])
                claims_tokens.append(segment_tokens[i])
            else:
                premises.append(segments[i])
                premises_tokens.append(segment_tokens[i])
        return major_claims, major_claims_tokens, claims, claims_tokens, premises, premises_tokens

    @staticmethod
    def _get_relation_args(idx1, idx2, token1, tokens2, text1, text2, cls_id, sep_id, max_len, pad_token):
        arg1 = token1[idx1]
        arg2 = tokens2[idx2]
        arg1_txt = text1[idx1]
        arg2_txt = text2[idx2]
        arg1 = utils.wrap_and_pad_tokens(inputs=arg1, prefix=cls_id, suffix=sep_id, seq_len=max_len, padding=pad_token)
        arg2 = utils.wrap_and_pad_tokens(inputs=arg2, prefix=cls_id, suffix=sep_id, seq_len=max_len, padding=pad_token)
        return arg1, arg1_txt, arg2, arg2_txt

    @staticmethod
    def _remove_padding(tokens, predictions, pad_token=0):
        new_tokens, new_predictions = [], []
        for idx, token in enumerate(tokens):
            if (token != pad_token) and (token != 101) and (token != 102):
                new_tokens.append(token)
                new_predictions.append(predictions[idx])
        return new_tokens, new_predictions

    def _get_relations(self, preds, kind="relations"):
        self.app_logger.debug("Getting predicted relations")
        relations = []
        other_int = self.relations_to_int["other"] if kind == "relations" else self.stance_to_int["other"]
        for pred in preds:
            args = pred[0]
            pred = pred[1]
            if pred == other_int:
                self.app_logger("No relation detected. Continuing...")
                continue
            arg1 = args[0]
            arg2 = args[1]
            relation = self.int_to_relations[pred] if kind == "relations" else self.int_to_stance[pred]
            relations.append((arg1, arg2, relation))
        return relations

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
