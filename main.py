import traceback

from classifiers import AduModel, RelationsModel
from pipeline import ArgumentMining
from data import DataLoader
from utils import AppConfig


def preprocess(app_config):
    logger = app_config.app_logger
    data_loader = DataLoader(app_config=app_config)
    logger.info("Creating CSV file in CONLL format for ADUs classification")
    data_loader.load_adus()
    logger.info("Creating CSV file in CONLL format for relations/stance classification")
    data_loader.load_relations()


def train(app_config):
    logger = app_config.app_logger
    models_to_train = app_config.properties["train"]["models"]
    if "adu" in models_to_train:
        logger.info("Training ADU classifier")
        adu_model = AduModel(app_config=app_config)
        adu_model.train()
        logger.info("ADU Training is finished!")
    if "rel" in models_to_train:
        logger.info("Training relations model")
        rel_model = RelationsModel(app_config=app_config, dev_csv=app_config.rel_dev_csv,
                                   train_csv=app_config.rel_train_csv, test_csv=app_config.rel_test_csv,
                                   base_path=app_config.rel_base_path, model_name="rel")
        rel_model.train()
        logger.info("Relations training finished!")
    if "stance" in models_to_train:
        logger.info("Training stance model")
        stance_model = RelationsModel(app_config=app_config, dev_csv=app_config.stance_dev_csv, model_name="stance",
                                      train_csv=app_config.stance_train_csv, test_csv=app_config.stance_test_csv,
                                      base_path=app_config.stance_base_path)
        stance_model.train()
        logger.info("Stance training finished!")


def evaluate(app_config):
    arg_mining = ArgumentMining(app_config=app_config)
    arg_mining.run_pipeline()


def main():
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
        app_config.send_email(body="Argument mining pipeline finished successfully",
                              subject="Argument mining run: {}".format(app_config.run))
    except(BaseException, Exception):
        app_config.app_logger.error(traceback.format_exc())
        app_config.send_email(
            body="Argument mining pipeline finished with errors".format(traceback.format_exc(limit=100)),
            subject="Error in argument mining run: {}".format(app_config.run))
    finally:
        try:
            app_config.elastic_save.stop()
            app_config.elastic_retrieve.stop()
        except(BaseException, Exception):
            app_config.app_logger.error("Could not close ssh tunnels")
            exit(-1)


if __name__ == '__main__':
    main()
