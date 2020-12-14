import hashlib
import json
import logging
import os
import smtplib
import ssl
import uuid
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import environ, getcwd
from os import mkdir
from os.path import exists, join
from pathlib import Path

import torch
import yaml


def get_curr_path():
    curr_dir = Path(getcwd())
    parent = str(curr_dir.parent)
    curr_dir = str(curr_dir)
    return curr_dir if curr_dir.endswith("mining") else parent


def configure_device():
    if torch.cuda.is_available():
        devices = environ.get("CUDA_VISIBLE_DEVICES", 0)
        if type(devices) == str:
            devices = devices.split(",")
            device_name = "cuda:{}".format(devices[0].strip())
        else:
            device_name = "cuda:{}".format(devices)
    else:
        device_name = "cpu"
    return device_name


def get_base_path(path, base_name, hidden_size, layers, use_crf, optimizer, learning_rate, mini_batch_size):
    # Create a base path:
    embedding_names = 'bert-greek'
    base_path = "{}-".format(base_name) + '-'.join([
        str(embedding_names),
        'hs=' + str(hidden_size),
        'hl=' + str(layers),
        'crf=' + str(use_crf),
        "optmizer=" + optimizer,
        'lr=' + str(learning_rate),
        'bs=' + str(mini_batch_size)
    ])
    base_path = join(path, base_path)
    try:
        # os.mkdir(base_path, 0o755)
        os.makedirs(base_path)
    except (OSError, Exception):
        pass
    return base_path


def get_initial_json(name, text):
    hash_id = hashlib.md5(name.encode())
    return {
        "id": hash_id.hexdigest(),
        "link": "",
        "description": "",
        "date": "",
        "tags": [],
        "document_link": "",
        "publishedAt": "",
        "crawledAt": "",
        "domain": "",
        "netloc": "",
        "content": text,
        "annotations": {
            "ADUs": [],
            "Relations": []
        }
    }


def load_data(base_path, filename):
    filepath = join(base_path, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["data"]["documents"]


class AppConfig:

    def __init__(self, app_path):
        self.log_filename = 'logs_%s' % datetime.now().strftime('%Y%m%d-%H%M%S')
        self.documents_pickle = "documents.pkl"
        self.properties_file = "properties.yaml"
        self.example_properties = "example_properties.yaml"
        self.app_path = app_path
        self.app_path = app_path
        self.run = uuid.uuid4().hex
        self._create_paths()
        self._create_output_dirs()
        self.properties = {}
        self.app_logger = None
        self.device_name = ""
        self.data_file = None
        self._configure()

    def _configure(self):
        # logging
        self.app_logger = self._config_logger()
        self.app_logger.info("Run id: {}".format(self.run))

        # properties
        self.properties = self._load_properties()
        config = self.properties["config"]
        self._config_email(config=config)
        self._configure_base_paths_and_data_files()
        self.device_name = configure_device()

    def _config_logger(self):
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        program_logger = logging.getLogger(__name__)

        program_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler("{0}/{1}.log".format(self.logs_path, self.log_filename))
        file_handler.setFormatter(log_formatter)
        program_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        program_logger.addHandler(console_handler)
        return program_logger

    def _configure_base_paths_and_data_files(self):
        properties = self.properties["adu_model"]
        self.adu_base_path = get_base_path(path=self.output_path, base_name="adu_model",
                                           hidden_size=properties["hidden_size"],
                                           layers=properties["rnn_layers"],
                                           use_crf=properties["use_crf"], optimizer=properties["optimizer"],
                                           learning_rate=properties["learning_rate"],
                                           mini_batch_size=properties["mini_batch_size"])
        properties = self.properties["rel_model"]
        self.rel_base_path = get_base_path(path=self.output_path, base_name="rel_model",
                                           hidden_size=properties["hidden_size"], use_crf=properties["use_crf"],
                                           optimizer=properties["optimizer"], layers=properties["layers"],
                                           learning_rate=properties["learning_rate"],
                                           mini_batch_size=properties["mini_batch_size"])
        self.stance_base_path = get_base_path(path=self.output_path, base_name="stance_model",
                                              hidden_size=properties["hidden_size"],
                                              use_crf=properties["use_crf"],
                                              optimizer=properties["optimizer"], layers=properties["layers"],
                                              learning_rate=properties["learning_rate"],
                                              mini_batch_size=properties["mini_batch_size"])

        config = self.properties["config"]["adu_data"]

        self.eval_doc = config["eval_doc"]
        self.adu_train_csv = config["train_csv"]
        self.adu_dev_csv = config["dev_csv"]
        self.adu_test_csv = config["test_csv"]

        config = self.properties["config"]["rel_data"]
        self.rel_train_csv = config["train_csv"]
        self.rel_dev_csv = config["dev_csv"]
        self.rel_test_csv = config["test_csv"]

        config = self.properties["config"]["stance_data"]
        self.stance_train_csv = config["train_csv"]
        self.stance_dev_csv = config["dev_csv"]
        self.stance_test_csv = config["test_csv"]

    def _load_properties(self):
        path_to_properties = join(self.resources_path, self.properties_file)
        path_to_example_properties = join(self.resources_path, self.example_properties)
        final_path = path_to_properties if exists(path_to_properties) else path_to_example_properties
        with open(final_path, "r") as f:
            properties = yaml.safe_load(f.read())
        return properties

    def _config_email(self, config):
        config_email = config["email"]
        self.sender_email = config_email.get("sender", "skthemeli@gmail.com")
        self.receiver_email = config_email.get("receiver", "skthemeli@gmail.com")
        self.password = config_email["password"]

    def _create_paths(self):
        self.resources_path = join(self.app_path, "resources")
        self.output_path = join(self.app_path, "output")
        self.logs_path = join(self.output_path, "logs")
        self.model_path = join(self.output_path, "model")
        self.tensorboard_path = join(self.app_path, "runs")
        self.out_files_path = join(self.output_path, "output_files")

    def _create_output_dirs(self):
        if not exists(self.output_path):
            mkdir(self.output_path)
        if not exists(self.out_files_path):
            mkdir(self.out_files_path)
        if not exists(self.logs_path):
            mkdir(self.logs_path)
        if not exists(self.tensorboard_path):
            mkdir(self.tensorboard_path)
        if not exists(self.model_path):
            mkdir(self.model_path)
        if not exists(join(self.model_path, self.run)):
            mkdir(join(self.model_path, self.run))
        if not exists(join(self.tensorboard_path, self.run)):
            mkdir(join(self.tensorboard_path, self.run))

    def send_email(self, body, subject=None):
        if not subject:
            subject = "Argument mining run"

        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = subject

        # Add body to email
        message.attach(MIMEText(body, "plain"))
        filename = "{}.log".format(self.log_filename)
        # Open PDF file in binary mode
        with open(join(self.logs_path, filename), "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()

        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.receiver_email, text)

