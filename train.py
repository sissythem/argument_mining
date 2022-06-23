import argparse
import os
from os.path import join
import json
import shutil
import logging
import torch
import uuid

from src.utils import timestamp, lock_write_file
from collections import namedtuple
from src.adu_models import ADUModel
from src.pair_classifier_models import PairClassifierModel
from pipeline import setup_logging


def get_args(input_args=None):
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("type",
                        help="What type of model to train. Supported types are ADU sequential and pair classifier models.",
                        choices=["adu", "pair"])
    parser.add_argument("-device", help="Device to utilize", default="cpu")
    parser.add_argument("-n_gpu", help="Number of gpus", default=1)
    parser.add_argument("-train_data", help="Train data directory", default="data/adu")
    parser.add_argument("-test_data", help="Test data directory", default="data/adu")
    parser.add_argument("-train_args", help="Training arguments", default="""
        {
        "batch_size": 8,
        "num_epochs": 10,
        "strategy": "epoch",
        "eval_steps": 1
        }
        """
                        )

    parser.add_argument("--trained_model_path", help="Path to trained model")
    parser.add_argument("--model_output_path", help="Output path to store model after traininig")
    parser.add_argument("--model_version", help="Assigned model version")
    parser.add_argument("--status_file_path", help="File to mark completion")
    parser.add_argument("--add_timestamp", help="Whether to add timestamp info to output path", action="store_true",
                        default=False)
    if input_args is None:
        args = parser.parse_args()
    else:
        namespace = argparse.Namespace(**input_args)
        args = parser.parse_args(namespace=namespace)
    return args


def run(args):
    """
    Run the training process
    Args:
        args: Argument collection for training configuration
    """
    if args.status_file_path is not None:
        lock_write_file(args.status_file_path, "started")

    logfile_path = setup_logging(logging_folder=".")
    if isinstance(args, dict):
        args = namedtuple("args", args.keys())

    # i / o
    # model name / input path
    if args.trained_model_path:
        model_name = args.trained_model_path
        logging.info(f"Finetuning existing model at {model_name}")
    else:
        # default model
        model_name = "nlpaueb/bert-base-greek-uncased-v1"

    # output path
    model_output_path = args.model_output_path or f"{args.type}_trained_model"

    # whether to append timestamp
    if args.add_timestamp:
        model_output_path += "_" + timestamp()
    logging.info(f"Will save new model to {model_output_path}")

    # model class
    if args.type.lower() == "adu":
        model_class = ADUModel
    elif args.type.lower() == "pair":
        # peek in training data to figure out the labelset
        model_class = PairClassifierModel
    else:
        logging.error(f"Undefined training type {args.type}")
        exit(1)

    # instantiation
    model = model_class(model_name, device=args.device, output_folder=model_output_path)

    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    # launch training
    train_args = json.loads(args.train_args)
    model.train(train_datadir=args.train_data, test_datadir=args.test_data, model_output_path=model_output_path,
                training_arguments=train_args)
    moved_logfile_path = join(model_output_path, os.path.basename(logfile_path))
    shutil.move(logfile_path, moved_logfile_path)
    version = args.model_version or str(uuid.uuid4())
    with open(join(os.path.dirname(logfile_path), "version.txt"), "w") as f:
        f.write(str(version))
    logging.info(f"Training logs at {moved_logfile_path}")

    if args.status_file_path is not None:
        lock_write_file(args.status_file_path, "completed")


if __name__ == "__main__":
    run(get_args())
