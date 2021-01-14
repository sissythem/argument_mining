import threading
import hashlib
import logging
import os
import random
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
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from sshtunnel import SSHTunnelForwarder


def name_exceeds_bytes(name):
    return utf8len(name) >= 255


def utf8len(s):
    return len(s.encode('utf-8'))


def create_document_id(text):
    hash_id = hashlib.md5(text.encode())
    return hash_id.hexdigest()


def get_initial_json(name, text):
    return {
        "id": create_document_id(text=name),
        "title": name,
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


class AppConfig:

    def __init__(self):
        random.seed(2020)
        self.documents_pickle = "documents.pkl"
        self._configure()
        self.elastic_retrieve = ElasticSearchConfig(properties=self.properties["config"], elasticsearch="retrieve")
        self.elastic_save = ElasticSearchConfig(properties=self.properties["config"], elasticsearch="save")

    def _configure(self):
        self.run = uuid.uuid4().hex
        self._configure_device()
        self._create_paths()
        self.properties = self._load_properties()

        # logging
        self.app_logger = self._config_logger()
        self.app_logger.info("Run id: {}".format(self.run))

        # training data
        self._configure_training_data_and_model_path()

        # email
        config = self.properties["config"]
        self._config_email(config=config)
        self._configure_training_data_and_model_path()

    def _configure_device(self):
        if torch.cuda.is_available():
            devices = environ.get("CUDA_VISIBLE_DEVICES", 0)
            if type(devices) == str:
                devices = devices.split(",")
                self.device_name = "cuda:{}".format(devices[0].strip())
            else:
                self.device_name = "cuda:{}".format(devices)
        else:
            self.device_name = "cpu"

    def _config_logger(self):
        self.log_filename = 'logs_%s' % datetime.now().strftime('%Y%m%d-%H%M%S')
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        program_logger = logging.getLogger("flair")

        program_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler("{0}/{1}.log".format(self.logs_path, self.log_filename))
        file_handler.setFormatter(log_formatter)
        program_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        program_logger.addHandler(console_handler)
        return program_logger

    def _load_properties(self):
        self.properties_file = "properties.yaml"
        self.example_properties = "example_properties.yaml"
        path_to_properties = join(self.resources_path, self.properties_file)
        path_to_example_properties = join(self.resources_path, self.example_properties)
        final_path = path_to_properties if exists(path_to_properties) else path_to_example_properties
        with open(final_path, "r") as f:
            properties = yaml.safe_load(f.read())
        return properties

    def _create_paths(self):
        curr_dir = Path(getcwd())
        parent = str(curr_dir.parent)
        curr_dir = str(curr_dir)
        self.app_path = curr_dir if curr_dir.endswith("mining") else parent

        self.resources_path = join(self.app_path, "resources")
        self.output_path = join(self.app_path, "output")
        self.logs_path = join(self.output_path, "logs")
        self.model_path = join(self.output_path, "model")
        self.tensorboard_path = join(self.app_path, "runs")
        self.out_files_path = join(self.output_path, "output_files")
        self._create_output_dirs()

    def _create_output_dirs(self):
        if not exists(self.output_path):
            mkdir(self.output_path)
        if not exists(self.out_files_path):
            mkdir(self.out_files_path)
        if not exists(self.logs_path):
            mkdir(self.logs_path)
        if not exists(self.tensorboard_path):
            mkdir(self.tensorboard_path)

    def _get_base_path(self, base_name):
        # Create a base path:
        embedding_names = 'bert-greek'
        properties = self.properties["adu_model"] if base_name == "adu_model" else self.properties["rel_model"]
        layers = properties["rnn_layers"] if base_name == "adu_model" else properties["layers"]
        base_path = "{}-".format(base_name) + '-'.join([
            str(embedding_names),
            'hs=' + str(properties["hidden_size"]),
            'hl=' + str(layers),
            'crf=' + str(properties["use_crf"]),
            "optmizer=" + properties["optimizer"],
            'lr=' + str(properties["learning_rate"]),
            'bs=' + str(properties["mini_batch_size"])
        ])
        base_path = join(self.output_path, base_path)
        try:
            os.makedirs(base_path)
        except (OSError, Exception):
            pass
        return base_path

    def _configure_training_data_and_model_path(self):
        self.adu_base_path = self._get_base_path(base_name="adu_model")
        self.rel_base_path = self._get_base_path(base_name="rel_model")
        self.stance_base_path = self._get_base_path(base_name="stance_model")

        config = self.properties["config"]["adu_data"]
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

    def _config_email(self, config):
        config_email = config["email"]
        self.sender_email = config_email.get("sender", "skthemeli@gmail.com")
        self.receiver_email = config_email.get("receiver", "skthemeli@gmail.com")
        self.password = config_email["password"]

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


class ElasticSearchConfig:

    def __init__(self, properties, elasticsearch):
        properties = properties["elastic_save"] if elasticsearch == "save" else properties["elastic_retrieve"]
        self.username = properties["username"]
        self.password = properties["password"]
        self.host = properties["host"]
        self.port = properties["port"]
        self.ssh_port = properties["ssh"]["port"]
        self.ssh_username = properties["ssh"]["username"]
        self.ssh_key = properties["ssh"]["key_path"]
        self.connect = properties["connect"]
        self._init_elasticsearch_client()

    def _init_elasticsearch_client(self, timeout=60):
        if self.connect == "key":
            self._init_ssh_tunnel()
            self.elasticsearch_client = Elasticsearch([{
                'host': "localhost",
                'port': self.tunnel.local_bind_port,
                'http_auth': (self.username, self.password)
            }], timeout=timeout)
        else:
            self.elasticsearch_client = Elasticsearch([{
                'host': self.host,
                'port': self.port,
                'http_auth': (self.username, self.password)
            }], timeout=timeout)

    def _init_ssh_tunnel(self):
        self.tunnel = SSHTunnelForwarder(
            ssh_address=self.host,
            ssh_port=self.ssh_port,
            ssh_username=self.ssh_username,
            ssh_private_key=self.ssh_key,
            remote_bind_address=('127.0.0.1', self.port),
            compression=True
        )
        self.tunnel.start()

    def stop(self):
        del self.elasticsearch_client
        [t.close() for t in threading.enumerate() if t.__class__.__name__ == "Transport"]
        self.tunnel.stop()

    def truncate_elasticsearch(self):
        Search(using=self.elasticsearch_client, index="httpresponses").query("match_all").delete()
