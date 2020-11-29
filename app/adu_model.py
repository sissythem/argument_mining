import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
import torch.utils.data as torch_data
from sklearn.metrics import accuracy_score
from torchcrf import CRF
from transformers import BertForTokenClassification

import utils


class AduClassifier(pl.LightningModule):

    def __init__(self, app_config, encoded_labels, datasets):
        super(AduClassifier, self).__init__()
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.device_name = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.encoded_labels = encoded_labels
        self.num_labels = len(encoded_labels.keys())

        self.total_data = datasets["total_data"]
        self.total_labels = datasets["total_labels"]

        self.train_data = utils.convert_ndarray_to_tensor(datasets["train_data"], device_name=self.device_name)
        self.train_labels = utils.convert_ndarray_to_tensor(datasets["train_labels"], device_name=self.device_name)
        self.test_data = utils.convert_ndarray_to_tensor(datasets["test_data"], device_name=self.device_name)
        self.test_labels = utils.convert_ndarray_to_tensor(datasets["test_labels"], device_name=self.device_name)
        self.validation_data = utils.convert_ndarray_to_tensor(datasets["validation_data"],
                                                               device_name=self.device_name)
        self.validation_labels = utils.convert_ndarray_to_tensor(datasets["validation_labels"],
                                                                 device_name=self.device_name)

        self.validation_output = None
        self.validation_accuracies = []
        self.mean_validation_accuracy = 0.0
        self.test_output = []
        self.test_accuracies = []
        self.test_accuracy = 0.0

        self.bert_model = BertForTokenClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1',
                                                                     num_labels=self.num_labels).to(self.device_name)
        self.ff = FeedForward(properties=self.properties, logger=self.app_logger, device_name=self.device_name,
                              num_labels=self.num_labels)
        self.crf = CRFLayer(properties=self.properties, logger=self.app_logger, num_labels=self.num_labels,
                            device_name=self.device_name, encoded_labels=encoded_labels)

    def train_dataloader(self):
        self.app_logger.debug("Creating train DataLoader")
        train_dataset = ArgumentMiningDataset(data=self.train_data, labels=self.train_labels)
        return torch_data.DataLoader(train_dataset, batch_size=self.properties["model"]["batch_size"])

    def val_dataloader(self):
        self.app_logger.debug("Creating validation DataLoader")
        validation_dataset = ArgumentMiningDataset(data=self.validation_data, labels=self.validation_labels)
        return torch_data.DataLoader(validation_dataset, batch_size=self.properties["model"]["batch_size"])

    def test_dataloader(self):
        self.app_logger.debug("Creating test DataLoader")
        test_dataset = ArgumentMiningDataset(data=self.test_data, labels=self.test_labels)
        return torch_data.DataLoader(test_dataset, batch_size=self.properties["model"]["batch_size"])

    def forward(self, tokens, labels=None):
        # in lightning, forward defines the prediction/inference actions
        self.app_logger.debug("Start forward")
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
        self.app_logger.debug("Training step loss: {}".format(loss))
        logs = {"train_loss": loss}
        # output = torch.reshape(output["sequence_of_tags"], (-1, 7)).to(self.device_name)
        output = output["sequence_of_tags"]
        y = y.flatten()
        y_true = y.to("cpu")
        y_true = y_true.numpy()
        y_pred = np.asarray(output)
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
        # loss = self.loss_function(logits=output, true_labels=y)
        # output = torch.argmax(output, dim=1)
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
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
        self.app_logger.debug("Validation step loss: {}".format(loss))
        logs = {"train_loss": loss}
        # self.validation_output = torch.reshape(self.validation_output, (-1, 7)).to(self.device_name)
        y = y.flatten()
        # loss = self.loss_function(logits=self.validation_output, true_labels=y)
        # output = torch.argmax(self.validation_output, dim=1)
        y_true = y.to("cpu")
        y_true = y_true.numpy()
        y_pred = np.asarray(self.validation_output)
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.validation_accuracies.append(accuracy)
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
        self.app_logger.debug("Test step loss: {}".format(loss))
        logs = {"train_loss": loss}
        y = y.flatten()
        # self.test_output = torch.reshape(self.test_output, (-1, 7)).to(self.device_name)
        # loss = self.loss_function(logits=self.test_output, true_labels=y)
        y_true = y.to("cpu")
        y_true = y_true.numpy()
        y_pred = np.asarray(predictions)
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.test_accuracies.append(accuracy)
        correct = (y_true == y_pred).sum()
        total = y.shape[0]

        # save output of testing
        self.test_output.append(y_pred)

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
        self.app_logger.debug("CRF sequence output: {}".format(sequence_of_tags))
        loss = -1 * log_likelihood  # Log likelihood is not normalized (It is not divided by the batch size).
        return {"loss": loss, "sequence_of_tags": sequence_of_tags}

    def decode(self, logits, mask):
        return self.crf.decode(logits, mask)


class FeedForward(torch.nn.Module):
    properties = None
    app_logger = None
    num_labels = 0

    def __init__(self, properties, logger, device_name, num_labels):
        super(FeedForward, self).__init__()
        self.properties = properties
        self.app_logger = logger
        self.device_name = device_name
        self.input_size = 768
        self.ff = torch.nn.Linear(self.input_size, num_labels).to(device_name)

    def forward(self, x):
        hidden = self.ff(x)
        activation_function = self._get_activation_function()
        res = activation_function(hidden)
        output = f.dropout(res)
        output = f.softmax(output, dim=2)
        return output

    def _get_activation_function(self):
        activation_func_name = self.properties["model"]["activation_function"]
        if activation_func_name == "relu":
            act_function = f.relu
        elif activation_func_name == "sigmoid":
            act_function = torch.sigmoid
        elif activation_func_name == "tanh":
            act_function = torch.tanh
        elif activation_func_name == "leaky_relu":
            act_function = f.leaky_relu
        else:
            # ReLU activations by default
            act_function = f.relu
        return act_function
