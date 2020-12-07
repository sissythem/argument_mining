import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
import torch.utils.data as torch_data
from sklearn.metrics import accuracy_score
from torchcrf import CRF
from transformers import BertForTokenClassification
from models.ff_model import FeedForward


class AduClassifier(pl.LightningModule):

    def __init__(self, app_config, encoded_labels, datasets):
        super(AduClassifier, self).__init__()
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.device_name = app_config.device_name
        self.encoded_labels = encoded_labels
        self.num_labels = len(encoded_labels.keys())

        self.total_data = datasets["total_data"]
        self.total_labels = datasets["total_labels"]

        self.train_data = torch.from_numpy(datasets["train_data"])
        self.train_labels = torch.from_numpy(datasets["train_labels"])
        self.test_data = torch.from_numpy(datasets["test_data"])
        self.test_labels = torch.from_numpy(datasets["test_labels"])
        self.validation_data = torch.from_numpy(datasets["validation_data"])
        self.validation_labels = torch.from_numpy(datasets["validation_labels"])

        self.validation_output = None
        self.validation_accuracies = []
        self.mean_validation_accuracy = 0.0
        self.test_output = []
        self.test_accuracies = []
        self.test_accuracy = 0.0

        self.bert_model = BertForTokenClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1',
                                                                     num_labels=self.num_labels).to(self.device_name)
        self.ff = FeedForward(properties=self.properties, logger=self.app_logger, device_name=self.device_name,
                              num_labels=self.num_labels, softmax_dim=2)
        self.crf = CRFLayer(properties=self.properties, logger=self.app_logger, num_labels=self.num_labels,
                            device_name=self.device_name, encoded_labels=encoded_labels)

    def train_dataloader(self):
        self.app_logger.debug("Creating train DataLoader")
        train_dataset = ArgumentMiningDataset(data=self.train_data, labels=self.train_labels)
        return torch_data.DataLoader(train_dataset, shuffle=True, batch_size=self.properties["model"]["batch_size"])

    def val_dataloader(self):
        self.app_logger.debug("Creating validation DataLoader")
        validation_dataset = ArgumentMiningDataset(data=self.validation_data, labels=self.validation_labels)
        return torch_data.DataLoader(validation_dataset, shuffle=True,
                                     batch_size=self.properties["model"]["batch_size"])

    def test_dataloader(self):
        self.app_logger.debug("Creating test DataLoader")
        test_dataset = ArgumentMiningDataset(data=self.test_data, labels=self.test_labels)
        return torch_data.DataLoader(test_dataset, shuffle=True, batch_size=self.properties["model"]["batch_size"])

    def forward(self, tokens, labels=None):
        self.app_logger.debug("Start forward")
        tokens = tokens.to(self.device_name)
        if labels is not None:
            labels = labels.to(self.device_name)
        bert_output = self.bert_model(input_ids=tokens, output_hidden_states=True)
        logits = bert_output[0]
        self.app_logger.debug("Bert output shape: {}".format(logits.shape))
        embeddings = bert_output[1][-1]
        ff_output = self.ff.forward(embeddings)
        self.app_logger.debug("FF output shape: {}".format(ff_output.shape))
        mask = tokens != self.properties["preprocessing"]["pad_token"]
        if labels is not None:
            output = self.crf.forward(logits=ff_output, labels=labels, mask=mask)
        else:
            output = self.crf.decode(ff_output, mask=mask)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.squeeze().to(self.device_name)
        y = y.squeeze().to(self.device_name)
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
            y = y.reshape(1, y.shape[0])
        self.app_logger.debug("Batch idx: {}".format(batch_idx))
        self.app_logger.debug("Input shape: {}".format(x.shape))
        self.app_logger.debug("Labels shape: {}".format(y.shape))
        output = self.forward(tokens=x, labels=y)
        loss = output["loss"]
        logs = {"train_loss": loss}
        self.app_logger.debug("Training step loss: {}".format(loss))
        output = output["sequence_of_tags"]
        y = y.flatten()
        y_true = y.to("cpu")
        y_true = y_true.numpy()
        y_pred = np.asarray(output)
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        correct = (y_true == y_pred).sum()
        total = y.shape[0]
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().to(self.device_name)
        y = y.squeeze().to(self.device_name)
        output = self.forward(tokens=x, labels=y)
        self.validation_output = output["sequence_of_tags"]
        loss = output["loss"]
        logs = {"val_loss": loss}
        self.app_logger.debug("Validation step loss: {}".format(loss))
        y = y.flatten()
        y_true = y.to("cpu")
        y_true = y_true.numpy()
        y_pred = np.asarray(self.validation_output)
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.validation_accuracies.append(accuracy)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        correct = (y_true == y_pred).sum()
        total = y.shape[0]
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().to(self.device_name)
        y = y.squeeze().to(self.device_name)
        output = self.forward(tokens=x, labels=y)
        predictions = output["sequence_of_tags"]
        loss = output["loss"]
        logs = {"test_loss": loss}
        self.app_logger.debug("Test step loss: {}".format(loss))
        y = y.flatten()
        y_true = y.to("cpu")
        y_true = y_true.numpy()
        y_pred = np.asarray(predictions)
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.test_accuracies.append(accuracy)
        # save output of testing
        self.test_output.append(y_pred)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        correct = (y_true == y_pred).sum()
        total = y.shape[0]
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
        return batch_dictionary

    def loss_function(self, logits, true_labels):
        loss_function_name = self.properties["model"]["loss"]
        if loss_function_name == "cross_entropy":
            loss = f.cross_entropy(input=logits, target=true_labels)
            # add more loss functions
        else:
            loss = f.cross_entropy(input=logits, target=true_labels)
        return loss

    def configure_optimizers(self):
        optimizer_name = self.properties["model"]["optimizer"]
        learning_rate = self.properties.get("learning_rate", 1e-3)
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


class ArgumentMiningDataset(torch_data.Dataset):
    data = None
    labels = None

    def __init__(self, data, labels):
        super(ArgumentMiningDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CRFLayer(torch.nn.Module):

    def __init__(self, properties, logger, device_name, num_labels, encoded_labels):
        super(CRFLayer, self).__init__()
        self.num_labels = num_labels
        self.properties = properties
        self.app_logger = logger
        self.device_name = device_name
        self.encoded_labels = encoded_labels
        self.crf = CRF(num_tags=num_labels, batch_first=True).to(self.device_name)

    def forward(self, logits, labels, mask):
        log_likelihood = self.crf.forward(emissions=logits, tags=labels, mask=mask)
        sequence_of_tags = self.crf.decode(logits)
        loss = -1 * log_likelihood  # Log likelihood is not normalized (It is not divided by the batch size).
        return {"loss": loss, "sequence_of_tags": sequence_of_tags}

    def decode(self, logits, mask):
        return self.crf.decode(logits, mask)
