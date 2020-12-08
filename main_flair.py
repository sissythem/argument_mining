import random
from os.path import join

from torch.optim.adam import Adam

from flair_impl import preprocessing as prep
from flair_impl import utils
from flair_impl import flair_simple as fl
from base.config import AppConfig


def main():
    random.seed(2020)
    properties = utils.get_properties()
    curr_dir = utils.get_curr_path()
    app_config = AppConfig(app_path=curr_dir)
    output_path = join(curr_dir, "output")
    data_folder = join(curr_dir, "resources")
    base_path = utils.get_base_path(path=output_path, hidden_size=properties["hidden_size"],
                                    rnn_layers=properties["rnn_layers"],
                                    use_crf=properties["use_crf"], optimizer=Adam,
                                    learning_rate=properties["learning_rate"],
                                    mini_batch_size=properties["mini_batch_size"])
    do_prep = properties["do_prep"]
    if do_prep:
        prep.preprocess_adus()
    fl.adu_train(properties=properties, data_folder=data_folder, base_path=base_path)


if __name__ == '__main__':
    main()
