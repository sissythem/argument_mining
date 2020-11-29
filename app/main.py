import pickle
import traceback
from datetime import datetime
from os import getcwd
from os.path import join

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit

from adu_model import AduClassifier
from relations_model import RelationsClassifier
from config import AppConfig
from data import DataLoader, DataPreprocessor


def adu_classification(app_config, sentences, labels, lbl_dict):
    logger = app_config.app_logger
    properties = app_config.properties
    logger.debug("Preparing train and test datasets")
    test_size = properties["validation"]["test_size"]
    folds = properties["validation"]["folds"]
    skip_test = True if test_size == 0.0 else False
    tokens = np.asarray([d.numpy().flatten() for d in sentences])
    labels = np.asarray([d.numpy().flatten() for d in labels])
    skip_validation = True if folds == 0 else False
    if not skip_test:
        train_data, test_data, train_labels, test_labels = train_test_split(tokens, labels, test_size=test_size,
                                                                            random_state=0)

        # splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        # train_ix ,test_ix = splitter.split(tokens, labels)
        # train_data, train_labels = (tokens[train_ix], labels[train_ix])
        # test_data, test_labels = (tokens[test_ix], labels[test_ix])

    elif skip_test and skip_validation:
        train_data, test_data, train_labels, test_labels = sentences, sentences, labels, labels
    else:
        train_data, train_labels = sentences, labels
        test_data, test_labels = [], []
    if not skip_validation:
        k_fold = KFold(n_splits=folds)
        classifiers = []
        trainers = []
        fold_counter = 0
        for train_index, test_index in k_fold.split(train_data, train_labels):
            fold_counter += 1
            logger.debug("Fold counter: {}".format(fold_counter))
            print(train_index, test_index)

            x_train_fold = train_data[train_index]
            y_train_fold = train_labels[train_index]
            x_validation_fold = train_data[test_index]
            y_validation_fold = train_labels[test_index]

            logger.debug("Train data shape: {}".format(x_train_fold.shape))
            logger.debug("Train labels shape: {}".format(y_train_fold.shape))
            logger.debug("Validation data shape: {}".format(x_validation_fold.shape))
            logger.debug("Validation labels shape: {}".format(y_validation_fold.shape))

            datasets = {
                "total_data": sentences,
                "total_labels": sentences,
                "train_data": x_train_fold,
                "train_labels": y_train_fold,
                "test_data": test_data,
                "test_labels": test_labels,
                "validation_data": x_validation_fold,
                "validation_labels": y_validation_fold
            }
            classifier = AduClassifier(app_config=app_config, encoded_labels=lbl_dict, datasets=datasets)
            epochs = properties["model"].get("epochs", 100)
            checkpoint_callback = ModelCheckpoint(
                filepath=join(app_config.model_path, app_config.run),
                verbose=True,
                monitor="train_loss",
                mode='min'
            )
            path = join(app_config.tensorboard_path, app_config.run)
            tb_logger = TensorBoardLogger(path, name="argument_mining")
            trainer = Trainer(max_epochs=epochs, min_epochs=1, checkpoint_callback=True, logger=tb_logger,
                              callbacks=[EarlyStopping(monitor="train_loss"), ProgressBar(), checkpoint_callback])
            trainer.fit(classifier)
            classifier.mean_validation_accuracy = np.mean(classifier.validation_accuracies)
            classifiers.append(classifier)
            trainers.append(trainer)
        for classifier, trainer in zip(classifiers, trainers):
            logger.debug("Testing classifier {}".format(classifiers.index(classifier)))
            trainer.test(classifier)
            classifier.test_accuracy = np.mean(classifier.test_accuracies)
        timestamp = datetime.now()
        models_file = "adu_classifiers_data_{}".format(timestamp)
        with open(join(app_config.output_path, models_file), "wb") as f:
            pickle.dump(classifiers, f)
        return classifiers


def relations_classification(app_config, data, kind="relation"):
    input_data, labels, initial_data, lbl_dict = data["data"], data["labels"], data["initial_data"], \
                                                 data["encoded_labels"]
    logger = app_config.app_logger
    properties = app_config.properties
    logger.debug("Preparing train and test datasets")
    test_size = properties["validation"]["test_size"]
    folds = properties["validation"]["folds"]
    skip_test = True if test_size == 0.0 else False
    tokens = np.asarray([d.numpy().flatten() for d in input_data])
    labels = np.asarray(labels, dtype=np.int64)
    if "sampling" in properties:
        smpl = properties["sampling"]
        print(f"OOF -- sampling to the first {smpl} data for testing purposes")
        tokens = tokens[:smpl]
        labels = labels[:smpl]
    logger.debug(f"Tokens shape: {tokens.shape}")
    skip_validation = True if folds == 0 else False
    if not skip_test:
        train_data, test_data, train_labels, test_labels = train_test_split(tokens, labels, test_size=test_size,
                                                                            random_state=0)
        # splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        # train_ix ,test_ix = splitter.split(tokens, labels)
        # train_data, train_labels = (tokens[train_ix], labels[train_ix])
        # test_data, test_labels = (tokens[test_ix], labels[test_ix])

    elif skip_test and skip_validation:
        train_data, test_data, train_labels, test_labels = data, data, labels, labels
    else:
        train_data, train_labels = data, labels
        test_data, test_labels = [], []
    if not skip_validation:
        k_fold = StratifiedKFold(n_splits=folds)
        classifiers = []
        trainers = []
        fold_counter = 0
        for train_index, test_index in k_fold.split(train_data, train_labels):
            fold_counter += 1
            logger.debug("Fold counter: {}".format(fold_counter))
            logger.debug(f"Trainval idx shapes: {train_index.shape, test_index.shape}")

            x_train_fold = train_data[train_index]
            y_train_fold = train_labels[train_index]
            x_validation_fold = train_data[test_index]
            y_validation_fold = train_labels[test_index]

            logger.debug("Train data shape: {}".format(x_train_fold.shape))
            logger.debug("Train labels shape: {}".format(y_train_fold.shape))
            logger.debug("Validation data shape: {}".format(x_validation_fold.shape))
            logger.debug("Validation labels shape: {}".format(y_validation_fold.shape))

            datasets = {
                "total_data": data,
                "total_labels": data,
                "train_data": x_train_fold,
                "train_labels": y_train_fold,
                "test_data": test_data,
                "test_labels": test_labels,
                "validation_data": x_validation_fold,
                "validation_labels": y_validation_fold
            }

            classifier = RelationsClassifier(app_config=app_config, encoded_labels=lbl_dict, datasets=datasets)
            epochs = properties["model"].get("epochs", 100)
            checkpoint_callback = ModelCheckpoint(
                filepath=join(app_config.model_path, app_config.run),
                verbose=True,
                monitor="train_loss",
                mode='min'
            )
            path = join(app_config.tensorboard_path, app_config.run)
            tb_logger = TensorBoardLogger(path, name="argument_mining")
            trainer = Trainer(max_epochs=epochs, min_epochs=1, checkpoint_callback=True, logger=tb_logger,
                              callbacks=[EarlyStopping(monitor="train_loss"), ProgressBar(), checkpoint_callback])
            trainer.fit(classifier)
            classifier.mean_validation_accuracy = np.mean(classifier.validation_accuracies)
            classifiers.append(classifier)
            trainers.append(trainer)
        for classifier, trainer in zip(classifiers, trainers):
            logger.debug("Testing classifier {}".format(classifiers.index(classifier)))
            trainer.test(classifier)
            classifier.test_accuracy = np.mean(classifier.test_accuracies)
        timestamp = datetime.now()
        models_file = "{}_classifiers_data_{}".format(kind, timestamp)
        with open(join(app_config.output_path, models_file + ".relational"), "wb") as f:
            pickle.dump(classifiers, f)
        return classifiers


def main():
    app_path = join(getcwd(), "app") if getcwd().endswith("argument_mining") else getcwd()
    app_config = AppConfig(app_path=app_path)
    app_config.configure()
    logger = app_config.app_logger
    try:
        logger.info("Loading data")
        data_loader = DataLoader(app_config=app_config)
        documents = data_loader.load_data()
        logger.info("Data are loaded!")
        logger.info("Preprocessing data")
        data_preprocessor = DataPreprocessor(app_config=app_config)
        sentences, labels, lbl_dict = data_preprocessor.preprocess(documents)
        logger.info("Preprocessing finished")

        # classifiers = adu_classification(app_config=app_config, sentences=sentences, labels=labels, lbl_dict=lbl_dict)
        # test_accuracies = [c.test_accuracy for c in classifiers]
        # max_accuracy = max(test_accuracies)
        # app_config.send_email(body="Run finished with test accuracy: {}".format(max_accuracy))

        # all_data = data_preprocessor.preprocess_relations(documents)
        # relations_classification(app_config=app_config, data=all_data["relation"])
        # relations_classification(app_config=app_config, data=all_data["stance"])

        cl_path = "/home/sthemeli/Έγγραφα/argument-mining/app/output/classifiers_data_2020-11-26 18:20:03.371906"
        build_output([d.content for d in documents], cl_path)

    except(Exception, BaseException) as e:
        logger.error("Error occurred: {}".format(traceback.format_exc()))
        app_config.send_email(body="An error occurred during the run!{}".format(traceback.format_exc()))

def select_best_classifier(classifiers):
    # sort by mean test acc.
    c = sorted(classifiers, key=lambda x: x.test_accuracy)
    return c[-1]


def detect_segments(preds, labels_dict):
    lbls = sorted(list(labels_dict.keys()))
    # starting segment labels
    start_lbls = [l for l in lbls if l.startswith("B")

    for d, doc_preds in enumerate(preds):
        for s, sent_preds in enumerate(doc_preds):
            # TODO
            # how do we handle cases of predicted tokens that make no sense?
            # e.g. B-majorclaim, I-premise
            pass
            
from transformers import BertTokenizer

def build_output(texts, adu_training_output):
    import data, pickle, torch
    with open(adu_training_output, "rb") as f:
        print("Loading outputs from", f.name)
        cl = pickle.load(f)

    device = "cuda:1"
    clf = select_best_classifier(cl)
    clf.to(device)
    seqlen = clf.properties["preprocessing"]["max_len"]
    pad_token = clf.properties["preprocessing"]["pad_token"]
    tok = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

    # token classification
    print("Doing token classification")
    preds = []
    for text in texts:
        preds.append([])
        sentences = data.tokenize_sentences(text)
        for s, sentence in enumerate(sentences):
            if len(sentence) > seqlen:
                print(f"(!!!) Sentence length #{s}:  {len(sentence)} but max len is {seqlen}")
            tokens = tok(sentence, is_split_into_words=True, add_special_tokens=True, padding="max_length", truncation=True, max_length=seqlen)["input_ids"]
            tokens = torch.LongTensor(tokens).unsqueeze(0).to(clf.device)
            predictions = clf.forward(tokens)
            preds[-1].append(predictions)
    breakpoint()
    segments = detect_segments(preds, clf.encoded_labels)
    with open("tok_classif.pkl", "wb") as f:
        pickle.dump((preds, clf), f)
    print()


if __name__ == '__main__':
    main()
    # build_output(cl_path)
