import pickle
from datetime import datetime
from os.path import join

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split, KFold

from models.adu_model import AduClassifier
from models.relations_model import RelationsClassifier


class ArgumentMiningTrainer:

    def __init__(self, app_config, data, labels, lbl_dict, test=False, val=False):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        test_size = self.properties["validation"]["test_size"]
        self.test_size = test_size
        self.num_folds = self.properties["validation"]["folds"]
        self.data = data
        self.labels = labels
        self.lbl_dict = lbl_dict
        if test:
            self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(data, labels,
                                                                                                    test_size=test_size,
                                                                                                    random_state=0)
        else:
            self.train_data, self.test_data, self.train_labels, self.test_labels = data, data, labels, labels
        self.folds = []
        if val:
            k_fold = KFold(n_splits=self.num_folds)
            self.classifiers = []
            self.trainers = []
            for train_index, test_index in k_fold.split(self.train_data, self.train_labels):
                x_train_fold = self.train_data[train_index]
                y_train_fold = self.train_labels[train_index]
                x_validation_fold = self.train_data[test_index]
                y_validation_fold = self.train_labels[test_index]
                fold = {"train_data": x_train_fold, "train_labels": y_train_fold, "validation_data": x_validation_fold,
                        "validation_labels": y_validation_fold}
                self.folds.append(fold)
        else:
            fold = {"train_data": self.train_data, "train_labels": self.train_labels, "validation_data": self.train_data,
                    "validation_labels": self.train_labels}
            self.folds.append(fold)

    def train(self, kind="adu"):
        fold_counter = 0

        for fold in self.folds:
            fold_counter += 1
            self.app_logger.debug("Fold counter: {}".format(fold_counter))

            datasets = {
                "total_data": self.data,
                "total_labels": self.labels,
                "train_data": fold["train_data"],
                "train_labels": fold["train_labels"],
                "test_data": self.test_data,
                "test_labels": self.test_labels,
                "validation_data": fold["validation_data"],
                "validation_labels": fold["validation_labels"]
            }
            if kind == "adu":
                classifier = AduClassifier(app_config=self.app_config, encoded_labels=self.lbl_dict, datasets=datasets)
            else:
                classifier = RelationsClassifier(app_config=self.app_config, encoded_labels=self.lbl_dict,
                                                 datasets=datasets)
            epochs = self.properties["model"].get("epochs", 100)
            checkpoint_callback = ModelCheckpoint(
                filepath=join(self.app_config.model_path, self.app_config.run),
                verbose=True,
                monitor="train_loss",
                mode='min'
            )
            path = join(self.app_config.tensorboard_path, self.app_config.run)
            tb_logger = TensorBoardLogger(path, name="argument_mining")
            trainer = Trainer(max_epochs=epochs, min_epochs=1, checkpoint_callback=True, logger=tb_logger,
                              callbacks=[EarlyStopping(monitor="train_loss"), ProgressBar(), checkpoint_callback])
            trainer.fit(classifier)
            classifier.mean_validation_accuracy = np.mean(classifier.validation_accuracies)
            self.classifiers.append(classifier)
            self.trainers.append(trainer)

    def test(self):
        for classifier, trainer in zip(self.classifiers, self.trainers):
            self.app_logger.debug("Testing classifier {}".format(self.classifiers.index(classifier)))
            trainer.test(classifier)
            classifier.test_accuracy = np.mean(classifier.test_accuracies)
        timestamp = datetime.now()
        models_file = "classifiers_{}".format(timestamp)
        with open(join(self.app_config.output_path, models_file), "wb") as f:
            pickle.dump(self.classifiers, f)
        return models_file
