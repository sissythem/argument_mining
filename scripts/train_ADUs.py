import argparse
import os
import torch
from src.utils import timestamp

from src.adu_models import ADUModel

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

model = ADUModel(model_name)

model.train(train_datadir=args.data_dir, test_datadir=args.data_dir, model_output_path=model_output_path, batch_size=2)