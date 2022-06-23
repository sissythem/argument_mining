from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollator, AutoModelForSequenceClassification
import torch
import os
from transformers import EarlyStoppingCallback, ProgressCallback
from datasets import load_metric
from src import dataset_utils
from src.utils import set_seed
from src.abstract_model import Model
import random
import argparse
import logging
import numpy as np


# from src.utils import *


class PairClassifierModel(Model):
    name = "pair_classifier"

    def get_null_label(self):
        return "other"

    def configure_labels(self, label_list):
        logging.info(f"Configuring pair classifier with {len(label_list)} labels: {label_list}")
        if self.label_list is not None:
            # if already configured, just verify the same set exists
            assert set(label_list) == set(self.label_list), "Asked to reconfigure with a different set of labels!"
            return
        self.label_list = label_list
        self.num_labels = len(self.label_list)
        self.label_encoding_dict = {k: v for v, k in enumerate(self.label_list)}
        self.index_encoding_dict = {v: k for v, k in enumerate(self.label_list)}
        self.index_encoding_dict[-100] = ""
        self.instantiate_model(can_delay=False)

    def instantiate_model(self, can_delay=True):
        if self.model is not None:
            # already defined
            return
        if self.num_labels is None:
            if not can_delay:
                raise ValueError(f"Cannot delay model instantiation and the number of labels is unknown!")
            logging.info(f"Delaying creating {self.model_name_or_path} model until the number of labels is known")
            return
        logging.info(f"Instantiating model to device: [{self.device}]")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        num_labels=self.num_labels).to(self.device)

    def __init__(self, model_name_or_path: str, label_list: list = None, num_labels=None, seed=2022, max_length=256,
                 device="cpu", output_folder=""):
        """
        Instantiate a new model
        :param model_name:
        :return:
        """
        super().__init__(output_folder)
        self.model_name_or_path = model_name_or_path
        self.model = self.num_labels = self.label_list = None
        self.device = device

        if label_list:
            self.configure_labels(label_list)
            if num_labels is not None:
                assert len(label_list) == num_labels, "Inconsistent number of labels!"
            self.num_labels = num_labels
        self.max_length = max_length
        set_seed(seed)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length=max_length)
        except:
            raise ValueError(f"Failed to instantiate huggingface tokenizer from path/ID: {model_name_or_path}")
        self.instantiate_model(can_delay=True)

    def predict(self, seq1, seq2):
        tokens = seq1 + " [SEP] " + seq2
        tokenized_inputs = self.tokenizer(tokens,  # padding="max_length", truncation=True,
                                          return_tensors="pt"
                                          ).to(self.device)

        outputs = self.model(**tokenized_inputs)
        logits = outputs.logits.detach().squeeze().cpu()
        score, pred = torch.softmax(logits, dim=0).max(axis=0)
        return float(score), self.index_encoding_dict[int(pred)]

    def get_traintest_data(self, train_data, test_data):

        logging.info("Getting datasets")
        train_dataset = dataset_utils.get_torch_dataset_from_path(train_data, dataset_utils.read_pairwise_csv)
        test_dataset = dataset_utils.get_torch_dataset_from_path(test_data, dataset_utils.read_pairwise_csv)

        train_data_labels = set(train_dataset["label"])
        self.configure_labels(train_data_labels)

        logging.info("Tokenizing")
        train_tokenized_datasets = train_dataset.map(self._tokenize_and_align_labels, batched=True).remove_columns(
            ["text_pair", "_"])
        test_tokenized_datasets = test_dataset.map(self._tokenize_and_align_labels, batched=True).remove_columns(
            ["text_pair", "_"])

        return train_tokenized_datasets, test_tokenized_datasets

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["text_pair"], padding="max_length", truncation=True,
                                          is_split_into_words=False, max_length=self.max_length)

        labels = [self.label_encoding_dict[l] for l in examples.data['label']]
        tokenized_inputs['label'] = labels
        # return {"input_ids": tokenized_inputs, "labels": labels}
        return tokenized_inputs

    def train(self, train_datadir, test_datadir, model_output_path: str, training_arguments: dict):

        train_data, test_data = self.get_traintest_data(train_datadir, test_datadir)

        training_args = self.get_training_arguments(training_arguments, len(train_data), len(test_data))

        # data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            # data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_arguments.get("early_stopping", 5)), ]
        )

        logging.info("Beginning training")
        trainer.train()
        logging.info("Beginning evaluation")
        trainer.evaluate()
        logging.info(f"Saving to {model_output_path}")
        trainer.save_model(model_output_path)

    def compute_metrics(self, p):
        # logging.info(flush=True)
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)

        metric = load_metric("f1")
        results = metric.compute(predictions=predictions, references=labels, average="macro")
        results["accuracy"] = load_metric("accuracy").compute(predictions=predictions, references=labels)
        return results

    def get_argmax(self, scores):
        maxidx = np.argmax(scores)
        return (self.label_list + ["other"])[maxidx], scores[maxidx]


class DummyPairClassifierModel(PairClassifierModel):
    def __init__(self, label_list):
        self.label_list = label_list

    def predict(self, tokens1, tokens2):
        nlabels = len(self.label_list)
        vec = np.random.random(nlabels)
        vec = vec / vec.sum()
        score, amx = vec.max(), vec.argmax()
        label = self.label_list[amx]
        return score, label


class RelationClassifier(PairClassifierModel):
    def __init__(self, model_name_or_path, device):
        super(RelationClassifier, self).__init__(model_name_or_path, label_list=["support", "attack", "other"],
                                                 device=device)


class StanceClassifier(PairClassifierModel):
    def __init__(self, model_name_or_path, device):
        super(StanceClassifier, self).__init__(model_name_or_path, label_list=["for", "against"], device=device)
