import random

from base import utils
from base.config import FlairConfig
from data import flair_preprocessing as prep
from models.flair_models import AduModel, RelationsModel, ArgumentMining


def main():
    random.seed(2020)
    curr_dir = utils.get_curr_path()
    app_config = FlairConfig(app_path=curr_dir)
    logger = app_config.app_logger
    properties = app_config.properties
    do_prep = properties["do_prep"]
    if do_prep:
        logger.info("Creating CSV file in CONLL format for ADUs classification")
        # TODO uncomment below line
        # prep.adu_preprocess(app_config=app_config)
        logger.info("Creating CSV file in CONLL format for relations classification")
        prep.preprocess_relations(app_config=app_config)
        logger.info("Creating CSV file in CONLL format for stance classification")
    adu_model = AduModel(app_config=app_config)
    rel_model = RelationsModel(app_config=app_config, dev_csv=app_config.rel_dev_csv,
                               train_csv=app_config.rel_train_csv, test_csv=app_config.rel_test_csv,
                               eval_doc=app_config.eval_doc, base_path=app_config.rel_base_path, model_name="rel")
    stance_model = RelationsModel(app_config=app_config, dev_csv=app_config.stance_dev_csv, model_name="stance",
                                  train_csv=app_config.stance_train_csv, test_csv=app_config.stance_test_csv,
                                  eval_doc=app_config.eval_doc, base_path=app_config.stance_base_path)
    if "train" in properties["tasks"]:
        logger.info("Training ADU classifier")
        adu_model.train()
        logger.info("ADU Training is finished!")
        logger.info("Training relations model")
        rel_model.train()
        logger.info("Relations training finished!")
        logger.info("Training stance model")
        stance_model.train()
        logger.info("Stance training finished!")
    if "eval" in properties["tasks"]:
        logger.info("Creating json output files with ADU & relation predictions")
        arg_mining = ArgumentMining(app_config=app_config, adu_model=adu_model, rel_model=rel_model,
                                    stance_model=stance_model)
        arg_mining.predict()
        logger.info("Evaluation is finished!")


if __name__ == '__main__':
    main()
