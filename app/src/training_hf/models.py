from transformers import AutoModelForTokenClassification, AutoConfig
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForTokenClassification, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification

from src.utils.config import AppConfig


class TransformerModel:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.logger = app_config.app_logger

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # flatten
        labels = np.reshape(labels, (-1,))
        predictions = np.reshape(predictions, (-1,))

        res = {"macro_f1": f1_score(labels, predictions, average="macro"),
               "micro_f1": f1_score(labels, predictions, average="micro"),
               "accuracy": accuracy_score(labels, predictions)}
        return res

    def train(self, model_id, tokenizer, num_labels, train_dset, eval_dset, seqlen, batch_size,
              eval_step_period, lr, epochs):
        cfg = AutoConfig.from_pretrained(model_id, num_labels=num_labels)
        model = AutoModelForTokenClassification.from_pretrained(model_id, config=cfg)
        dc = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, max_length=seqlen)
        training_args = TrainingArguments("test_trainer", per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          evaluation_strategy="steps", eval_steps=eval_step_period, do_train=True,
                                          do_eval=True, learning_rate=lr, metric_for_best_model="accuracy",
                                          load_best_model_at_end=True, report_to=["tensorboard"],
                                          logging_dir=self.app_config.logs_path,
                                          greater_is_better=True, save_strategy="steps", save_total_limit=5,
                                          num_train_epochs=epochs, logging_strategy="steps", logging_steps=100)
        trainer = Trainer(
            model=model,
            data_collator=dc,
            args=training_args,
            train_dataset=train_dset,
            eval_dataset=eval_dset,
            compute_metrics=self.compute_metrics)

        self.logger.info("Running training")
        trainer.train()
        print("Done training!")
