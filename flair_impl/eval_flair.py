from os import getcwd
from os.path import join
from pathlib import Path
from flair.models import SequenceTagger
import torch
from torch.optim.adam import Adam
from flair.data import Sentence
from flair_impl import utils

curr_dir = Path(getcwd())
curr_dir = str(curr_dir) if str(curr_dir).endswith("mining") else str(curr_dir.parent)
output_path = join(curr_dir, "output")
properties = utils.get_properties()
base_path = utils.get_base_path(path=output_path, hidden_size=properties["hidden_size"],
                                rnn_layers=properties["rnn_layers"],
                                use_crf=properties["use_crf"], optimizer=Adam,
                                learning_rate=properties["learning_rate"],
                                mini_batch_size=properties["mini_batch_size"])

# load the model you trained
model = SequenceTagger.load(base_path)

# create example sentence
sentence = Sentence('Γιατί όχι αεροδρόμιο στο Καστέλλι')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
