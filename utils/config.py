import base64
import json
import logging
import os
import random
import smtplib
import ssl
import threading
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

import requests
import torch
import yaml
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from sshtunnel import SSHTunnelForwarder


class AppConfig:
    """
    Class to initialize the application properties
    """

    def __init__(self):
        """
        Constructor for the Argument Mining application configuration. It configures the names of the project's folders
        and data files, reads the dynamic configuration from a yaml file, initializes two Elasticsearch clients, i.e.
        one for the social observatory and one for the debatelab elasticsearch and sets some configuration
        """
        random.seed(2020)
        self.documents_pickle = "documents.pkl"
        self._configure()
        self.elastic_retrieve = ElasticSearchConfig(properties=self.properties["config"], elasticsearch="retrieve")
        self.elastic_save = ElasticSearchConfig(properties=self.properties["config"], elasticsearch="save")

    def _configure(self):
        """
        Initialize the application configuration
        """
        self.run = uuid.uuid4().hex
        self._configure_device()
        self._create_paths()
        self.properties = self._load_properties()

        # logging
        self.app_logger = self._config_logger()
        self.app_logger.info(f"Run id: {self.run}")

        # training data
        self._configure_training_data_and_model_path()

        # email
        config = self.properties["config"]
        self._config_email(config=config)
        self._configure_training_data_and_model_path()

    def _configure_device(self):
        """
        Reads the environmental variable CUDA_VISIBLE_DEVICES in order to initialize the device to be used
        in the training
        """
        if torch.cuda.is_available():
            devices = environ.get("CUDA_VISIBLE_DEVICES", 0)
            if type(devices) == str:
                devices = devices.split(",")
                self.device_name = f"cuda:{devices[0].strip()}"
            else:
                self.device_name = f"cuda:{devices}"
        else:
            self.device_name = "cpu"

    def _config_logger(self):
        """
        Configures the application logger

        Returns
            logger: the initialized logger
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_filename = f"logs_{timestamp}.log"
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        program_logger = logging.getLogger("flair")

        program_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(f"{self.logs_path}/{self.log_filename}")
        file_handler.setFormatter(log_formatter)
        program_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        program_logger.addHandler(console_handler)
        return program_logger

    def _load_properties(self):
        """
        Loads the configuration file from the resources folder

        Returns
            dict: the application properties
        """
        self.properties_file = "properties.yaml"
        self.example_properties = "example_properties.yaml"
        path_to_properties = join(self.resources_path, self.properties_file)
        path_to_example_properties = join(self.resources_path, self.example_properties)
        final_path = path_to_properties if exists(path_to_properties) else path_to_example_properties
        with open(final_path, "r") as f:
            properties = yaml.safe_load(f.read())
        return properties

    def _create_paths(self):
        """
        Creates the various paths to application directories, e.g. output, resources, logs etc
        """
        curr_dir = Path(getcwd())
        parent = str(curr_dir.parent)
        curr_dir = str(curr_dir)
        self.app_path = curr_dir if curr_dir.endswith("mining") else parent

        self.resources_path = join(self.app_path, "resources")
        self.output_path = join(self.app_path, "output")
        self.logs_path = join(self.output_path, "logs")
        self.model_path = join(self.output_path, "models")
        self.output_files = join(self.output_path, "output_files")
        self.tensorboard_path = join(self.app_path, "runs")
        self._create_output_dirs()

    def _create_output_dirs(self):
        """
        Create missing directories (e.g. logs, output etc)
        """
        if not exists(self.output_path):
            mkdir(self.output_path)
        if not exists(self.logs_path):
            mkdir(self.logs_path)
        if not exists(self.model_path):
            mkdir(self.model_path)
        if not exists(self.tensorboard_path):
            mkdir(self.tensorboard_path)
        if not exists(self.output_files):
            mkdir(self.output_files)
        if not exists(join(self.resources_path, "data")):
            mkdir(join(self.resources_path, "data"))
        if not exists(join(self.resources_path, "results")):
            mkdir(join(self.resources_path, "results"))

    def _get_base_path(self, base_name):
        """
        Create the base full path to the directory where each model will be saved

        Args
            base_name (str): the name of the model

        Returns
            str: the path to the directory of the model
        """
        # Create a base path:
        embedding_names = 'bert-greek'
        if base_name == "adu_model":
            properties = self.properties["adu_model"]
        elif base_name == "sim_model":
            properties = self.properties["sim_model"]
        else:
            properties = self.properties["rel_model"]
        bert_kind = properties.get("bert_kind", "aueb")
        layers = properties["rnn_layers"] if base_name == "adu_model" else properties["layers"]
        base_path = f"{base_name}-" + '-'.join([
            str(embedding_names),
            "bert=" + bert_kind,
            'hs=' + str(properties["hidden_size"]),
            'hl=' + str(layers),
            'crf=' + str(properties["use_crf"]),
            "optmizer=" + properties["optimizer"],
            'lr=' + str(properties["learning_rate"]),
            'bs=' + str(properties["mini_batch_size"])
        ])
        base_path = join(self.model_path, base_path)
        try:
            os.makedirs(base_path)
        except (OSError, Exception):
            pass
        return base_path

    def _configure_training_data_and_model_path(self):
        """
        Reads the application properties to find for each model the train, test and dev datasets
        """
        self.adu_base_path = self._get_base_path(base_name="adu_model")
        self.rel_base_path = self._get_base_path(base_name="rel_model")
        self.stance_base_path = self._get_base_path(base_name="stance_model")
        self.sim_base_path = self._get_base_path(base_name="sim_model")

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

        config = self.properties["config"]["sim_data"]
        self.sim_train_csv = config["train_csv"]
        self.sim_dev_csv = config["dev_csv"]
        self.sim_test_csv = config["test_csv"]

    def _config_email(self, config):
        """
        Email configuration in order to get notification when the program has finished

        Args
            config (dict): configuration parameters for email
        """
        config_email = config["email"]
        self.sender_email = config_email.get("sender", "skthemeli@gmail.com")
        self.receiver_email = config_email.get("receiver", "skthemeli@gmail.com")
        self.password = config_email["password"]

    def update_last_retrieve_date(self):
        # update last search date
        properties = self.properties
        if properties["eval"]["retrieve"] == "date":
            properties["eval"]["last_date"] = datetime.now()
            with open(join(self.resources_path, self.properties_file), "w") as f:
                yaml.dump(properties, f)

    def notify_ics(self, ids_list, kind="arg_mining"):
        """
        Function to notify ICS API for any updates in the Elasticsearch. Uses different API endpoints based on the
        kind parameter. The possible notifications are: argument mining updates for new documents, clustering updates
        for cross-document relations

        Args
            | ids_list: list of ids in the Elasticsearch (documents or relations)
            | kind: the kind of update, possible values --> arg_mining, clustering
        """
        properties = self.properties["eval"]["notify"]
        # TODO url based on kind. Credentials?
        url = properties["url_arg_mining"] if kind == "arg_mining" else properties["url_clustering"]
        username = properties["username"]
        password = properties["password"]
        data = {"properties": {"delivery_mode": 2}, "routing_key": "dlabqueue", "payload": json.dumps(ids_list),
                "payload_encoding": "string"}
        creds = f"{username}:{password}"
        creds_bytes = creds.encode("ascii")
        base64_bytes = base64.b64encode(creds_bytes)
        base64_msg = base64_bytes.decode("ascii")
        headers = {"Content-Type": "application/json", "Authorization": f"Basic {base64_msg}"}
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                self.app_logger.info("Request to ICS was successful!")
            else:
                self.app_logger.error(
                    f"Request to ICS failed with status code: {response.status_code} and message:{response.text}")
        except(BaseException, Exception) as e:
            self.app_logger.error(f"Request to ICS failed: {e}")

    def send_email(self, body, subject=None):
        """
        Function to send a notification email upon completion of the program

        Args
            body (str): the body message
            subject (str): the subject of the email
        """
        if not subject:
            subject = "Argument mining run"

        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = subject

        # Add body to email
        message.attach(MIMEText(body, "plain"))
        # Open PDF file in binary mode
        with open(join(self.logs_path, self.log_filename), "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {self.log_filename}",
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
    """
    Class to configure an Elasticsearch Client
    """

    def __init__(self, properties, elasticsearch):
        """
        Constructor of the ElasticSearchConfig class

        Args
            | properties (dict): a dictionary with the configuration parameters
            | elasticsearch (str): valid values are save or retrieve in order to determine which parameters to use
            The elastic_save parameters are configurations for the debatelab elasticsearch while the elastic_retrieve
            properties are the configurations, credentials etc of the socialobservatory elasticsearch.
        """
        properties = properties["elastic_save"] if elasticsearch == "save" else properties["elastic_retrieve"]
        self.username = properties["username"]
        self.password = properties["password"]
        self.host = properties["host"]
        self.port = properties["port"]
        self.ssh_port = properties["ssh"]["port"]
        self.ssh_username = properties["ssh"]["username"]
        self.ssh_password = properties["ssh"]["password"]
        self.ssh_key = properties["ssh"]["key_path"]
        self.connect = properties["connect"]
        self._init_ssh_tunnel()
        self._init_elasticsearch_client()

    def _init_elasticsearch_client(self, timeout=60):
        """
        Initialization of the Elasticsearch client

        Args
            timeout (int): optional parameter to define the timeout for the connection
        """
        self.elasticsearch_client = Elasticsearch([{
            'host': "localhost",
            'port': self.tunnel.local_bind_port,
            'http_auth': (self.username, self.password)
        }], timeout=timeout)

    def _init_ssh_tunnel(self):
        """
        Initialization of ssh tunnel connection in order to use it to create the Elasticsearch client
        """
        if self.connect == "key":
            self.tunnel = SSHTunnelForwarder(
                ssh_address=self.host,
                ssh_port=self.ssh_port,
                ssh_username=self.ssh_username,
                ssh_private_key=self.ssh_key,
                remote_bind_address=('127.0.0.1', self.port),
                compression=True
            )
        else:
            self.tunnel = SSHTunnelForwarder(
                ssh_address=self.host,
                ssh_port=self.ssh_port,
                ssh_username=self.ssh_username,
                ssh_password=self.ssh_password,
                remote_bind_address=('127.0.0.1', self.port),
                compression=True
            )
        self.tunnel.start()

    def stop(self):
        """
        Stop the ssh tunneling
        """
        del self.elasticsearch_client
        [t.close() for t in threading.enumerate() if t.__class__.__name__ == "Transport"]
        self.tunnel.stop()

    def truncate_elasticsearch(self):
        """
        Delete all entries in the elasticsearch
        """
        Search(using=self.elasticsearch_client, index="httpresponses").query("match_all").delete()

    def retrieve_documents(self, previous_date=None, retrieve_kind="file"):
        if retrieve_kind == "file":
            file_path = join(getcwd(), "resources", "kasteli_34_urls.txt")
            # read the list of urls from the file:
            with open(file_path, "r") as f:
                urls = [line.rstrip() for line in f]
            search_articles = Search(using=self.elasticsearch_client, index='articles').filter('terms', link=urls)
        else:
            date_range = {'gt': previous_date, 'lte': datetime.now()}
            search_articles = Search(using=self.elasticsearch_client, index='articles').filter('range', date=date_range)
        documents = []
        for hit in search_articles.scan():
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            if not document["content"].startswith(document["title"]):
                document["content"] = document["title"] + "\r\n\r\n" + document["content"]
            documents.append(document)
        return documents

    def save_document(self, document):
        self.elasticsearch_client.index(index='debatelab', ignore=400, refresh=True,
                                        doc_type='docket', id=document["id"],
                                        body=document)

    def save_relation(self, relation):
        # TODO change index
        self.elasticsearch_client.index(index='debatelab', ignore=400, refresh=True,
                                        doc_type='docket', id=relation["id"],
                                        body=relation)
