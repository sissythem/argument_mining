import argparse
import torch
import pandas as pd
from src.utils import *
import os
from os.path import join
from src.pair_classifier_models import PairClassifierModel

parser = argparse.ArgumentParser()
parser.add_argument("-n_gpu", help="Number of gpus", default=1)
parser.add_argument("-data_dir", help="Data directory", default="data/adu")
parser.add_argument("--model_path", help="Path to trained model", default='trained_model')
args = parser.parse_args()

if os.path.isdir(args.model_path):
    model_path = args.model_path
    model_name = model_path
    print(f"Finetuning existing model at {model_path}")
    model_output_path = "_".join(model_path.split("_")[:-2]) + "_" + timestamp()
else:
    # default model
    model_name = "nlpaueb/bert-base-greek-uncased-v1"
    # append with timestamp
    model_output_path = f"{args.model_path}_{timestamp()}"

print(f"Will save new model to {model_output_path}")
print(f"Is CUDA available: {torch.cuda.is_available()}")


train_data = pd.read_csv(join(args.data_dir, "train.csv"), sep="\t", header=None)
labels = list(set(train_data[1].values.tolist()))
model = PairClassifierModel(model_name, label_list=labels)

train = join(args.data_dir, "train.csv")
test = join(args.data_dir, "test.csv")
model.train(train_datadir=train, test_datadir=test, model_output_path=model_output_path, batch_size=1)
