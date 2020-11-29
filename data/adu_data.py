import pickle
from os.path import join, exists

import torch
from transformers import BertTokenizer


class DataPreprocessor:

    def __init__(self, app_config):
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.resources_folder = app_config.resources_path
        self.pickle_file = app_config.pickle_sentences_filename
        self.labels_to_int = {}
        self.int_to_labels = {}
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
            self.app_logger.debug("Processing sentence {} and labels {}".format(sentence, lbls))
            other_label = self.properties["preprocessing"]["other_label"]
            tokens, new_labels = self._tokenize_input(sentence=sentence, labels=lbls, other_label=other_label)
            final_sentences.append(tokens)
            final_labels.append(new_labels)
        max_len = self.properties["preprocessing"]["max_len"]
        pad_token = self.properties["preprocessing"]["pad_token"]
        final_sentences, final_labels = self._add_padding(tokens=final_sentences, labels=final_labels, max_len=max_len,
                                                          pad_token=pad_token)
        output = (final_sentences, final_labels, self.labels_to_int)
        with open(join(self.resources_folder, "adu_labels.pkl"), "wb") as f:
            pickle.dump(obj=(self.labels_to_int, self.int_to_labels), file=f)
        with open(join(self.resources_folder, self.pickle_file), "wb") as f:
            pickle.dump(obj=output, file=f)
        return output

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

    def _tokenize_input(self, sentence, labels, other_label="O"):
        tokens, new_labels = self._tokenize_sentence(
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

    def _tokenize_sentence(self, sentence, labels):
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
