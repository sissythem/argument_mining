import random

from base import utils
from base.config import FlairConfig
from data import flair_preprocessing as prep
from models.adu_model_flair import AduModel


def main():
    random.seed(2020)
    curr_dir = utils.get_curr_path()
    app_config = FlairConfig(app_path=curr_dir)
    properties = app_config.properties
    do_prep = properties["do_prep"]
    if do_prep:
        prep.adu_preprocess(app_config=app_config)
    adu_model = AduModel(app_config=app_config)
    if "train" in properties["tasks"]:
        adu_model.train()
    if "eval" in properties["tasks"]:
        adu_model.predict()


if __name__ == '__main__':
    main()
