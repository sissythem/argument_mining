import random

from base import utils
from base.config import FlairConfig
from data import flair_preprocessing as prep
from models.adu_model_flair import AduModel


def main():
    random.seed(2020)
    curr_dir = utils.get_curr_path()
    app_config = FlairConfig(app_path=curr_dir)
    logger = app_config.app_logger
    properties = app_config.properties
    do_prep = properties["do_prep"]
    if do_prep:
        logger.info("Creating CSV file in CONLL format for ADUs classification")
        prep.adu_preprocess(app_config=app_config)
    adu_model = AduModel(app_config=app_config)
    if "train" in properties["tasks"]:
        logger.info("Training ADU classifier")
        adu_model.train()
        logger.info("Training is finished!")
    if "eval" in properties["tasks"]:
        logger.info("Creating json output files with ADU & relation predictions")
        adu_model.predict()
        logger.info("Evaluation is finished!")


if __name__ == '__main__':
    main()
