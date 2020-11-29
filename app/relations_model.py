import pytorch_lightning as pl
import torch
import torch.nn.functional as f
import torch.utils.data as torch_data
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertModel

import utils


class RelationsDataset(torch_data.Dataset):
    data = None
    labels = None

    def __init__(self, data, labels):
        super(RelationsDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class RelationsClassifier(pl.LightningModule):

    def __init__(self, app_config, encoded_labels, datasets):
        super(RelationsClassifier, self).__init__()
        self.app_logger = app_config.app_logger
        self.properties = app_config.properties
        self.device_name = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.encoded_labels = encoded_labels
        self.num_labels = len(encoded_labels[0].keys())

        self.total_data = datasets["total_data"]
        self.total_labels = datasets["total_labels"]
        print("This is memory-hoggish for no reason -- TODO fix")
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

        # self.bert_model = BertForSequenceClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1',
        #                                                                 num_labels=self.num_labels).to(self.device_name)
        self.bert_model = BertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1').to(self.device_name)
        self.ff = FeedForward(properties=self.properties, logger=self.app_logger, device_name=self.device_name,
                              num_labels=self.num_labels)

    def train_dataloader(self):
        self.app_logger.debug("Creating train DataLoader")
        train_dataset = RelationsDataset(data=self.train_data, labels=self.train_labels)
        return torch_data.DataLoader(train_dataset, batch_size=self.properties["model"]["batch_size"])

    def val_dataloader(self):
        self.app_logger.debug("Creating validation DataLoader")
        validation_dataset = RelationsDataset(data=self.validation_data, labels=self.validation_labels)
        return torch_data.DataLoader(validation_dataset, batch_size=self.properties["model"]["batch_size"])

    def test_dataloader(self):
        self.app_logger.debug("Creating test DataLoader")
        test_dataset = RelationsDataset(data=self.test_data, labels=self.test_labels)
        return torch_data.DataLoader(test_dataset, batch_size=self.properties["model"]["batch_size"])

    def forward(self, tokens, labels):
        # in lightning, forward defines the prediction/inference actions
        self.app_logger.debug("Start forward")
        self.app_logger.debug("Start BERT training")
        bert_output = self.bert_model(input_ids=tokens, output_hidden_states=True)
        # candidates:
        # last hidden state of the model
        last_hidden = bert_output[0]
        # the pooler output, meant to be used for seq. classification
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        pooled_output = bert_output[1]
        # alternatively, you can use a combination of the last hidden states
        # e.g.: https://github.com/huggingface/transformers/issues/1328
        # all_hidden_states = some_func(bert_output[2])

        embeddings = pooled_output
        self.app_logger.debug("Bert output shape: {}".format(last_hidden.shape))
        self.app_logger.debug("Start FF training")
        ff_output = self.ff.forward(embeddings)
        return ff_output


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().to(self.device_name)
        y = y.squeeze().to(self.device_name)
        output = self.forward(tokens=x, labels=y)
        loss = self.loss_function(output, y)

        # save output of testing
        preds = torch.argmax(output, dim=1)
        self.test_output.append(preds)

        accuracy, num_correct = utils.get_accuracy_numcorrect(output, y)
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": {"test_loss": loss},
            # info to be used at epoch end
            "correct": num_correct,
            "total": len(y),
            "accuracy": accuracy
        }
        return batch_dictionary


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.squeeze().to(self.device_name)
        y = y.squeeze().to(self.device_name)
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        self.app_logger.debug("Batch idx: {}".format(batch_idx))
        self.app_logger.debug("Input shape: {}".format(x.shape))
        self.app_logger.debug("Labels shape: {}".format(y.shape))

        output = self.forward(tokens=x, labels=y)
        loss = self.loss_function(logits=output, true_labels=y)
        self.app_logger.debug("Training step loss: {}".format(loss))
        logs = {"train_loss": loss}

        accuracy, num_correct = utils.get_accuracy_numcorrect(output, y)

        total = y.shape[0]
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            # info to be used at epoch end
            "correct": num_correct,
            "total": total,
            "accuracy": accuracy
        }
        return batch_dictionary

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

    def loss_function(self, logits, true_labels):
        loss_function_name = self.properties["model"]["loss"]
        if loss_function_name == "cross_entropy":
            loss = f.cross_entropy(input=logits, target=true_labels)
            # add more loss functions
        else:
            loss = f.cross_entropy(input=logits, target=true_labels)
        return loss




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
        output = f.softmax(output, dim=1)
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
