import random
from collections import Counter, defaultdict
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, \
    TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, TokenClassificationPipeline
from transformers import EarlyStoppingCallback
from datasets import load_metric
from src import dataset_utils
from src.utils import tokenize_with_spans, set_seed, tokens_to_text, remove_accents
from src import utils
from src.abstract_model import Model
from os.path import join

import numpy as np


class DummySequenceModel:
    def __init__(self):
        self.label_list = [
            'O',  # Outside of an ADU
            'B-major_claim',  # Beginning of a Major Claim, right after another Major Claim
            'I-major_claim',  # Major Claim
            'B-claim',  # Beginning of a Claim right after another Claim
            'I-claim',  # Claim
            'B-premise',  # Beginning of a Premise right after another Premise
            'I-premise',  # Premise
        ]
        self.model = self

    def predict(self, tokens, original_text):
        raw_labels, labels = [], []
        for tok in tokens:
            val = random.random()
            if val < 0.5:
                val = random.random()
                if val < 0.3:
                    l = "major_claim"
                elif val < 0.6:
                    l = "claim"
                else:
                    l = "premise"
            else:
                l = "O"
            raw_labels.append(l)
        # put Bs

        prev = ""
        for i in range(len(raw_labels)):
            l = raw_labels[i]
            if l == "O":
                labels.append(l)
                prev = l
                continue
            if l == prev:
                labels.append("I-" + l)
                continue
            else:
                l = "B-" + raw_labels[i]
                labels.append(l)
            prev = raw_labels[i]

        mid = len(labels) // 2
        sz = 5
        labels[mid: mid + sz] = ["I-premise"] * sz
        labels[mid] = "B-premise"

        return labels, np.random.rand(len(labels))

    def _handle_multiple_major_claims(self, major_claims, adus):
        slist = list(sorted(major_claims, key=lambda x: int(x["starts"])))

        # first merge any contiguous mcs
        i = 0
        while i < len(slist) - 1:
            a = major_claims[i]
            b = major_claims[i + 1]
            if a['starts'] == b['ends']:
                # merge
                merged = {"id": f"{a['id']}_{b['id']}", "type": "major_claim",
                          "segment": a['segment'] + b['segment'],
                          "starts": min(a['starts']),
                          "confidence": str(np.mean([a['confidence'], b['confidence']]))
                          }
                slist = [slist[:i] + [merged] + slist[i + 2:]]
            else:
                i += 1

        # after merging, keep only the mc that has the highest confidence
        rng = list(range(len(major_claims)))
        maxconf_idx = max(zip(rng, major_claims),
                          key=lambda x: x[1]['confidence'])[0]
        resolved_major_claim = major_claims[maxconf_idx]
        # convert the rest to plain claims
        claims = []
        for i in rng:
            if i != maxconf_idx:
                cl = major_claims[i]
                cl["type"] = "claim"
                claims.append(cl)
        adus.extend(claims)
        return self._handle_adu_ids(adus, resolved_major_claim)

    def _handle_adu_ids(self, adus, major_claim):
        # insert the major claim
        major_claim["id"] = "T1"
        # set other adus to ids T2, T3, ...
        segment_counter = 2
        for adu in adus:
            adu["id"] = f"T{segment_counter}"
            segment_counter += 1
        adus = [major_claim] + adus
        return adus, segment_counter

    def _handle_missing_major_claim(self, adus, title):
        # set title as the major claim
        seg = {
            "id": "T1",
            "type": "major_claim",
            "starts": '0',
            "ends": str(len(title)),
            "segment": title,
            "confidence": 0.99
        }
        logging.info(
            f"Injecting document title: {title} as the fallback major claim.")
        # remove any adu with the same / overlapping segment content
        overlaps = [a for a in adus if a['segment'] == title or int(a['starts']) < int(seg['ends'])]
        if overlaps:
            logging.debug("Discarding ADUs for overlapping with inserted title major claim:")
            for o in overlaps:
                logging.debug(f"{o['id']} : {o['starts'], o['ends']} - {o['segment']}")
        adus = [a for a in adus if a not in overlaps]
        return self._handle_adu_ids(adus, seg)


class ADUModel(Model):
    name = "token_classifier"

    def __init__(self, model_name_or_path: str, seed=2022, max_length=128, device="cpu", output_folder=""):
        """
        Instantiate a new model
        :param model_name:
        :return: = 
        """
        super().__init__(output_folder)
        self.label_list = [
            'O',  # Outside of an ADU
            'B-major_claim',  # Beginning of a Major Claim, right after another Major Claim
            'I-major_claim',  # Major Claim
            'B-claim',  # Beginning of a Claim right after another Claim
            'I-claim',  # Claim
            'B-premise',  # Beginning of a Premise right after another Premise
            'I-premise',  # Premise
        ]
        self.label_encoding_dict = {k: v for v, k in enumerate(self.label_list)}
        self.index_encoding_dict = {v: k for v, k in enumerate(self.label_list)}
        self.max_length = max_length

        self.device = device
        set_seed(seed)

        # self.config = AutoConfig.from_pretrained(model_name_or_path, label2id=self.label_encoding_dict, id2label=self.index_encoding_dict)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, config=self.config)
        # self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=self.config)

        try:
            logging.info(f"Instantiating model to device: [{device}]")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length=max_length)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path,
                                                                         num_labels=len(self.label_list)).to(device)
        except Exception as ex:
            raise ValueError(
                f"Failed to instantiate huggingface model from path/ID: {model_name_or_path}, reason: {str(ex)}")

        self.pipeline = None
        self.index_encoding_dict[-100] = ""

    def _tokenize_and_align_labels(self, examples, task="ner"):
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(list(examples["tokens"]), padding="max_length", max_length=self.max_length,
                                          truncation=True,
                                          is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == '0':
                    label_ids.append(0)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_encoding_dict[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(self.label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_traintest_data(self, train_data_path, test_data_path):

        train_dataset = dataset_utils.get_torch_dataset_from_path(train_data_path, dataset_utils.read_adu_csv)
        test_dataset = dataset_utils.get_torch_dataset_from_path(test_data_path, dataset_utils.read_adu_csv)

        train_tokenized_datasets = train_dataset.map(self._tokenize_and_align_labels, batched=True)
        test_tokenized_datasets = test_dataset.map(self._tokenize_and_align_labels, batched=True)

        return train_tokenized_datasets, test_tokenized_datasets

    def train(self, train_datadir, test_datadir, model_output_path: str, training_arguments: dict):

        train_data, test_data = self.get_traintest_data(train_datadir, test_datadir)

        training_args = self.get_training_arguments(training_arguments, len(train_data), len(test_data))

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_arguments.get("early_stopping", 5))]
        )

        logging.info("Beginning training")
        trainer.train()
        logging.info("Beginning evaluation")
        trainer.evaluate()
        logging.info(f"Saving to {model_output_path}")
        trainer.save_model(model_output_path)

    def predict(self, tokens, original_text, best_effort=True):
        self.model.eval()
        # if self.pipeline is None:
        #     self.pipeline = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer, ignore_labels=[])
        tokenized_inputs = self.tokenizer(tokens,  # padding="max_length", max_length=self.max_length, truncation=True,
                                          padding="max_length",
                                          max_length=self.max_length,
                                          truncation=True,
                                          is_split_into_words=True, return_tensors="pt",
                                          return_special_tokens_mask=True,
                                          return_offsets_mapping=self.tokenizer.is_fast
                                          ).to(self.device)
        # get outputs, remove unexpected arg for special tokens
        unexpected_args = ("special_tokens_mask", "offset_mapping")
        outputs = self.model(**{k: v for (k, v) in tokenized_inputs.items() if k not in unexpected_args})
        logits = outputs.logits.detach().squeeze()
        scores, preds = torch.softmax(logits, dim=1).max(axis=1)
        # gather
        scores = scores.cpu()
        preds = preds.cpu()

        # how to deal with subword information:
        # option 1:
        # deal with token alignment with a hf pipeline
        # this applies the model to each token separately ignoring temporal dependencies;
        # predictions degenerate to "O" class for each
        # results = self.pipeline(tokens)

        # option 2:
        # this applies tokenization to subword components by the tokenizer, overriding our premade tokenization
        # results = self.pipeline(original_text)

        # option 3:
        # use the grouping mechanism in huggingface pipelines:
        # we can call the components separately setting the already tokenized words;
        # but for this we need the original text to automatically discover group subword pieces
        # and computed subword offsets have to arise from the entire text; now they arise from already tokenized
        # words
        # pipeline_inputs = tokenized_inputs
        # pipeline_inputs.logits = outputs.logits.detach()
        # pipeline_inputs["sentence"] = original_text
        # pipeline_inputs.logits = outputs.logits.detach()
        # fw = self.pipeline.forward(pipeline_inputs)
        # pipe_out = self.pipeline.postprocess(fw, ignore_labels=[])

        resolved_tokens = defaultdict(list)
        resolved_scores = defaultdict(list)
        resolved_preds = defaultdict(list)
        subword_prefix = getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None)
        # merge subword tokens by word ids
        for i, word_id in enumerate(tokenized_inputs.word_ids()):
            if word_id is None:
                continue
            score, pred, tok = scores[i], preds[i], tokenized_inputs.tokens()[i]
            tok = tok[len(subword_prefix):] if tok.startswith(subword_prefix) else tok
            resolved_scores[word_id].append(score)
            resolved_tokens[word_id].append(tok)
            resolved_preds[word_id].append(pred)

        resolved_tokens = {k: "".join(toks) for (k, toks) in resolved_tokens.items()}
        resolved_preds = {k: Counter(pred).most_common()[0][0] for (k, pred) in resolved_preds.items()}
        resolved_scores = {k: np.mean(sc) for (k, sc) in resolved_scores.items()}

        # to list
        keys = sorted(resolved_tokens.keys())
        assert len(keys) == len(resolved_tokens) == len(resolved_preds) == len(resolved_scores), "Length mismatch!"
        resolved_tokens = [resolved_tokens[k] for k in keys]
        resolved_preds = [resolved_preds[k] for k in keys]
        resolved_scores = [resolved_scores[k] for k in keys]

        # token normalization is not dependable, TODO improve consistency
        if not len(resolved_tokens) == len(tokens):
            # print("Resolved token error!")
            msg = f"HF {len(resolved_tokens)}-long tokenization cannot match the {len(tokens)}-long input tokens!: {tokens} -- {resolved_tokens}"
            if best_effort:
                logging.error(msg)
            else:
                raise ValueError(msg)
        # # do not compare UNKs
        # resolved_tokens = [r if r != "[UNK]" else t for (r, t) in zip(resolved_tokens, tokens)]
        # if resolved_tokens != [remove_accents(t.lower()) if t != "[UNK]" else "[UNK]" for t in tokens]:
        #     logging.error("Token mismatch!")
        #     print(resolved_tokens)
        #     print([remove_accents(t.lower()) for t in tokens])
        #     raise("Token mismatch.")
        labels = [self.index_encoding_dict[int(i)] for i in resolved_preds]
        return labels, resolved_scores

    def merge_predictions_by_subword_symbol(self, scores, tokenized_inputs, preds):
        # this method only relies on subword symbols, which are not always inserted (e.g. for single-word tokens sequence "efsyn.gr"
        # deprecated

        resolved_tokens, resolved_scores, resolved_preds = [], [], []

        subword_tokens = tokenized_inputs.encodings[0].tokens
        subword_prefix = getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None)

        for score, pred, tok in zip(scores, preds, subword_tokens):
            if tok in self.tokenizer.special_tokens_map.values():
                continue
            if tok.endswith(subword_prefix):
                # undefined
                assert False, "Double subword symbols!"
            if tok.startswith(subword_prefix):
                assert len(resolved_tokens) > 0, "Error in decoding subword units!"
                resolved_tokens[-1] += tok[len(subword_prefix):]
                resolved_scores[-1].append(score)
                # retain first pred
                # resolved_preds.append(pred)
            else:
                resolved_tokens.append(tok)
                resolved_scores.append([score])
                resolved_preds.append(pred)
        return resolved_scores, resolved_preds, resolved_tokens

    def compute_metrics(self, p):
        # print(flush=True)
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # print(predictions[0], labels[0], flush=True)
        # print(len(predictions[0]), len(labels[0]), flush=True)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric = load_metric("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)
        # print(results)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
