import traceback
from os.path import join

import numpy as np
import pandas as pd

from pipeline.debatelab import ArgumentMining
from training.models import AduModel, RelationsModel
from training.preprocessing import DataLoader
from utils.config import AppConfig


def error_analysis(path_to_resources):
    """
    Function to perform error analysis on the results. Saves the incorrect predictions into a file

    Args
        path_to_resources (str): the full path to the resources folder
    """
    path_to_results = join(path_to_resources, "results", "test.tsv")
    results = pd.read_csv(path_to_results, sep=" ", index_col=None, header=None, skip_blank_lines=False)
    df_list = np.split(results, results[results.isnull().all(1)].index)
    sentences = []
    for df in df_list:
        df = df[df[0].notna()]
        df[3] = np.where(df[1] == df[2], 0, 1)
        sentences.append(df)
    sentences_df = pd.concat(sentences)
    sentences_df.to_csv(join(path_to_resources, "results.csv"), sep="\t", index=False, header=False)
    error_sentences = []
    for sentence_df in sentences:
        if 1 in sentence_df[3].values:
            total_text = ""
            for index, row in sentence_df.iterrows():
                text, true_lbl, pred_lbl, diff = row
                total_text += f"{text} <{true_lbl}> " if diff == 0 else \
                    f"{text} <{true_lbl}> <{pred_lbl}> "
            print(total_text.strip())
            print("==============================================================================")
            error_sentences.append(total_text + "\n\n")
    with open(join(path_to_resources, "errors.txt"), "w") as f:
        f.writelines(error_sentences)


def preprocess(app_config):
    """
    Preprocess the data into CONLL format

    Args
        app_config (AppConfig): the application configuration
    """
    logger = app_config.app_logger
    data_loader = DataLoader(app_config=app_config)
    data_loader.load()
    logger.info("Creating CSV file in CONLL format for ADUs classification")
    data_loader.load_adus()
    logger.info("Creating CSV file in CONLL format for relations/stance classification")
    data_loader.load_relations()


def train(app_config):
    """
    Train the selected models. In the application properties, the models to be trained are indicated.

    Args
        app_config (AppConfig): the application configuration
    """
    logger = app_config.app_logger
    models_to_train = app_config.properties["train"]["models"]
    if "adu" in models_to_train:
        logger.info("Training ADU classifier")
        adu_model = AduModel(app_config=app_config)
        adu_model.train()
        logger.info("ADU Training is finished!")
    if "rel" in models_to_train:
        logger.info("Training relations model")
        rel_model = RelationsModel(app_config=app_config, model_name="rel")
        rel_model.train()
        logger.info("Relations training finished!")
    if "stance" in models_to_train:
        logger.info("Training stance model")
        stance_model = RelationsModel(app_config=app_config, model_name="stance")
        stance_model.train()
        logger.info("Stance training finished!")
    if "sim" in models_to_train:
        logger.info("Training argument similarity model")
        sim_model = RelationsModel(app_config=app_config, model_name="sim")
        sim_model.train()
        logger.info("Finished training similarity model")


def evaluate(app_config):
    """
    Execute the DebateLab pipeline

    Args
        app_config (AppConfig): the application configuration
    """
    arg_mining = ArgumentMining(app_config=app_config)
    arg_mining.run_pipeline()


def main():
    """
    The main function of the program. Initializes the AppConfig class to load the application properties and
    configurations and based on the tasks in the properties executes the necessary steps (preprocessing, training,
    DebateLab pipeline, error analysis)
    """
    app_config: AppConfig = AppConfig()
    try:
        properties = app_config.properties
        tasks = properties["tasks"]
        if "prep" in tasks:
            preprocess(app_config=app_config)
        if "train" in tasks:
            train(app_config=app_config)
        if "eval" in properties["tasks"]:
            evaluate(app_config=app_config)
        if "error" in properties["tasks"]:
            error_analysis(path_to_resources=app_config.resources_path)
        app_config.send_email(body="Argument mining pipeline finished successfully",
                              subject=f"Argument mining run: {app_config.run}")
    except(BaseException, Exception):
        app_config.app_logger.error(traceback.format_exc())
        app_config.send_email(
            body=f"Argument mining pipeline finished with errors {traceback.format_exc(limit=100)}",
            subject=f"Error in argument mining run: {app_config.run}")
    finally:
        try:
            app_config.elastic_save.stop()
            app_config.elastic_retrieve.stop()
        except(BaseException, Exception):
            app_config.app_logger.error("Could not close ssh tunnels")
            exit(-1)


if __name__ == '__main__':
    main()
