import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
import torch.utils.data as torch_data
from transformers import BertModel
from models.ff_model import FeedForward
from base import utils


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

        self.bert_model = BertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1').to(self.device_name)
        self.doc_embedding = "pooler"
        # get doc representation from the bert encoder output
        # candidates:
        if self.doc_embedding == "pooler":
            # the pooler output, meant to be used for seq. classification
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            # pooled_output = bert_output[1]
            self.bert_output_idx = 1
        elif self.doc_embedding == "last_hidden":
            # last hidden state of the model
            # last_hidden = bert_output[0]
            self.bert_output_idx = 0
        else:
            # alternatively, you can use a combination of the last hidden states
            # e.g.: https://github.com/huggingface/transformers/issues/1328
            # all_hidden_states = some_func(bert_output[2])
            print(f"Undefined doc-embedding acquisition type: {self.doc_embedding}")
            exit(1)

        self.ff = FeedForward(properties=self.properties, logger=self.app_logger, device_name=self.device_name,
                              num_labels=self.num_labels, softmax_dim=1)

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

    def forward(self, tokens, labels=None):
        # in lightning, forward defines the prediction/inference actions
        self.app_logger.debug("Start forward")
        self.app_logger.debug("Start BERT training")
        seq_len = self.properties["preprocessing"]["max_len"]
        pad_token = self.properties["preprocessing"]["pad_token"]
        in1 = tokens[:, 0, :]
        in2 = tokens[:, 1, :]
        inputs = []
        for i in range(self.properties["model"]["batch_size"]):
            input1 = in1[i, :].numpy()
            input2 = in2[i, :].numpy()
            input1 = input1[input1 != 0]
            input1 = input1[input1 != 101]
            input1 = input1[input1 != 102]
            input2 = input2[input2 != 0]
            input2 = input2[input2 != 101]
            input2 = input2[input2 != 102]
            input1 = torch.LongTensor(input1)
            input2 = torch.LongTensor(input2)
            input_tokens = torch.cat((input1, input2), 0)

            input_tokens = utils.wrap_and_pad_tokens(inputs=input_tokens, prefix=101, suffix=102, seq_len=seq_len,
                                                     padding=pad_token)
            inputs.append(input_tokens)
        inputs = torch.LongTensor(inputs).to(self.device_name)
        output = self.bert_model(input_ids=inputs, output_hidden_states=True)
        embeddings = output[self.bert_output_idx]

        self.app_logger.debug("Bert output shape: {}".format(embeddings.shape))
        self.app_logger.debug("Start FF training")
        ff_output = self.ff.forward(embeddings)
        return ff_output

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze()
        y = y.squeeze()
        output = self.forward(tokens=x, labels=y)
        loss = self.loss_function(output, y)
        # save output of testing
        preds = torch.argmax(output, dim=1)
        self.test_output.append(preds)
        accuracy, num_correct = self.get_accuracy_numcorrect(output, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.squeeze()
        y = y.squeeze()
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        self.app_logger.debug("Batch idx: {}".format(batch_idx))
        self.app_logger.debug("Input shape: {}".format(x.shape))
        self.app_logger.debug("Labels shape: {}".format(y.shape))
        output = self.forward(tokens=x, labels=y)
        loss = self.loss_function(logits=output, true_labels=y)
        self.app_logger.debug("Training step loss: {}".format(loss))
        accuracy, num_correct = self.get_accuracy_numcorrect(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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

    @staticmethod
    def get_accuracy_numcorrect(preds, y, is_proba=True):
        if is_proba:
            # NOTE: this is applicable only for single-label classification
            # for multiclass, you have to use a threshold, e.g. thresh=0.5:
            # predictions = (output > thresh)
            # and modify the code below for multiclass stats and metrics
            preds = torch.argmax(preds, dim=1)
        correct_boolean = (y == preds)
        accuracy = correct_boolean.float().mean().cpu()
        num_correct = correct_boolean.sum().cpu()
        return accuracy, num_correct
