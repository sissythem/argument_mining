import traceback
from os import getcwd
from os.path import join
import numpy as np

from base.config import AppConfig
from data.adu_data import DataPreprocessor as adudp
from data.relations_data import DataPreprocessor as reldp
from data.training_data import DataLoader
from models.training import ArgumentMiningTrainer
from models.arg_mining_pipeline import ArgumentMining


def load_documents(logger, app_config):
    logger.info("Loading data")
    data_loader = DataLoader(app_config=app_config)
    documents = data_loader.load()
    logger.info("Data are loaded!")
    return documents


def adu_preprocessing(documents, logger, app_config):
    logger.info("Preprocessing data")
    dp1 = adudp(app_config=app_config)
    sentences, labels, lbl_dict = dp1.preprocess(documents)
    sentences = np.asarray([d.numpy().flatten() for d in sentences])
    labels = np.asarray([d.numpy().flatten() for d in labels])
    logger.info("Preprocessing finished")
    return sentences, labels, lbl_dict


def adu_training(sentences, labels, lbl_dict, app_config):
    adu_trainer = ArgumentMiningTrainer(app_config, sentences, labels, lbl_dict)
    adu_trainer.train(kind="adu")
    return adu_trainer.test()


def relations_preprocessing(documents, app_config, kind="relations"):
    dp2 = reldp(app_config)
    all_data = dp2.preprocess(documents)
    if kind == "relations":
        data = all_data["relation"]
    else:
        data = all_data["stance"]
    input_data, labels, initial_data, lbl_dict = data["data"], data["labels"], data["initial_data"], \
                                                 data["encoded_labels"]
    sentences = np.asarray([(d[0].numpy().flatten(), d[1].numpy().flatten()) for d in input_data])
    labels = np.asarray(labels, dtype=np.int64)
    return sentences, labels, lbl_dict


def relations_training(app_config, data, labels, lbl_dict):
    relations_trainer = ArgumentMiningTrainer(app_config=app_config, data=data, labels=labels, lbl_dict=lbl_dict)
    relations_trainer.train(kind="relations")
    return relations_trainer.test()


def main():
    app_path = join(getcwd()) if getcwd().endswith("argument_mining") else getcwd()
    app_config = AppConfig(app_path=app_path)
    app_config.configure()
    logger = app_config.app_logger
    tasks = app_config.properties["tasks"]
    try:
        adu_model_file, relations_model_file, stance_model_file = app_config.properties["classifier_files"][0], \
                                                                  app_config.properties["classifier_files"][1], \
                                                                  app_config.properties["classifier_files"][2]
    except (KeyError, Exception):
        adu_model_file, relations_model_file, stance_model_file = "", "", ""
    try:
        documents = load_documents(logger=logger, app_config=app_config)
        if "train" in tasks:
            sentences, labels, lbl_dict = adu_preprocessing(documents=documents, logger=logger, app_config=app_config)
            adu_model_file = adu_training(sentences=sentences, labels=labels, lbl_dict=lbl_dict, app_config=app_config)
            # app_config.send_email(body="Finished ADU model training")
            logger.info("Preprocessing relations data")
            # rel_sentences, rel_labels, rel_lbl_dict = relations_preprocessing(documents=documents,
            #                                                                   app_config=app_config)
            logger.info("Preprocessing stance data")
            # stance_sentences, stance_lbls, stance_lbl_dict = relations_preprocessing(documents=documents,
            #                                                                          app_config=app_config,
            #                                                                          kind="stance")
            # relations_model_file = relations_training(app_config=app_config, data=rel_sentences, labels=rel_labels,
            #                                          lbl_dict=rel_lbl_dict)
            # app_config.send_email(body="Finished relations model training")
            # stance_model_file = relations_training(app_config=app_config, data=stance_sentences, labels=stance_lbls,
            #                                       lbl_dict=stance_lbl_dict)
            # app_config.send_email(body="Finished stance model training")
        if "exec" in tasks:
            arg_mining = ArgumentMining(app_config=app_config)
            arg_mining.load(adu_model_file_path=adu_model_file, relations_model_file_path=relations_model_file,
                            stance_model_file_path=stance_model_file)
            arg_mining.predict(documents=documents)
            app_config.send_email(body="Finished predicting input documents")
    except(Exception, BaseException) as e:
        logger.error("Error occurred: {}".format(traceback.format_exc()))
        app_config.send_email(body="An error occurred during the run!{}".format(traceback.format_exc()))


if __name__ == '__main__':
    main()
