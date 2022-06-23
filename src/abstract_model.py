import logging
from os.path import join

from transformers import TrainingArguments
import torch


class Model:
    name = "MODEL_NAME"

    def __init__(self, output_folder=""):
        self.output_folder = output_folder

    def empty_cuda_cache(self):
        torch.cuda.empty_cache()

    def get_training_arguments(self, training_arguments: dict, num_train_data: int, num_test_data: int):
        eval_save_strategy = training_arguments.get("strategy", "epoch")
        eval_steps = training_arguments.get("eval_steps", 2 if eval_save_strategy == "epoch" else 100)
        batch_size = training_arguments.get("batch_size", 8)
        num_epochs = training_arguments.get("num_epochs", 2)
        logging.info(f"""Using training args: 
        batch_size: {batch_size}
        num_epochs: {num_epochs}
        eval_steps {eval_steps}
        eval_save_strategy: {eval_save_strategy}
        """)
        total_steps = num_epochs * num_train_data // batch_size
        logging_period = min(total_steps // 10, 500)
        logging.info(f"Will log results with a period of {logging_period} steps")
        training_args = TrainingArguments(
            output_dir=join(self.output_folder, f"args_{self.name}"),
            evaluation_strategy=eval_save_strategy,
            save_strategy=eval_save_strategy,
            # evaluation_strategy="steps",
            eval_steps=eval_steps,  # Evaluation and Save happens every 50 steps
            save_total_limit=5,  # Only last 5 models are saved. Older ones are deleted.
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            push_to_hub=False,
            metric_for_best_model='f1',
            load_best_model_at_end=True,
            logging_steps=max(min(total_steps // 10, 500), 10)  # log at least 10 times per training
        )
        return training_args
