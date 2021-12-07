import base64
import json
import logging
import os
import random
import smtplib
import ssl
import threading
import uuid
import datetime as datetime_base
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import environ
from os import makedirs
from os.path import exists, join
from pathlib import Path
from typing import List, Dict, AnyStr

import requests
import torch
import yaml
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from sshtunnel import SSHTunnelForwarder

from src.utils.utils import normalize_newlines


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
        self._configure()

    def _configure(self):
        """
        Initialize the application configuration
        """
        random.seed(2020)
        # run ID for this execution
        self.run: AnyStr = uuid.uuid4().hex

        # select device to run the models: gpu, cpu
        self._configure_device()
        # create necessary folders
        self._create_paths()
        # load properties file
        self.properties: Dict = self._load_properties()

        # logging
        self.app_logger: logging.Logger = self._config_logger()
        self.app_logger.info(f"Run id: {self.run}")

        self.adu_base_path: str = self._get_base_path(base_name="adu")
        self.rel_base_path: str = self._get_base_path(base_name="rel")
        self.stance_base_path: str = self._get_base_path(base_name="stance")
        self.sim_base_path: str = self._get_base_path(base_name="sim")

        # config elasticsearch
        self._config_elastic_search()

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

    def _config_logger(self) -> logging.Logger:
        """
        Configures the application logger

        Returns
            logger: the initialized logger
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_filename = f"logs_{timestamp}.log"
        log_formatter = logging.Formatter(
            '%(asctime)s,%(msecs)d %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s')
        program_logger = logging.getLogger("flair")

        program_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            f"{self.logs_path}/{self.log_filename}")
        file_handler.setFormatter(log_formatter)
        [program_logger.removeHandler(h) for h in program_logger.handlers]
        program_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        program_logger.addHandler(console_handler)
        return program_logger

    def _load_properties(self) -> Dict:
        """
        Loads the configuration file from the resources folder

        Returns
            dict: the application properties
        """
        self.properties_file = "properties.yaml"
        print("Loading properties from", self.properties_file)
        self.example_properties = "example_properties.yaml"
        path_to_properties = join(self.resources_path, self.properties_file)
        path_to_example_properties = join(
            self.resources_path, self.example_properties)
        final_path = path_to_properties if exists(
            path_to_properties) else path_to_example_properties
        with open(final_path, "r") as f:
            properties = yaml.safe_load(f.read())
        return properties

    def _create_paths(self):
        """
        Creates the various paths to application directories, e.g. output, resources, logs etc
        """
        self.app_path = Path(__file__).parent.parent.parent
        self.resources_path: AnyStr = join(self.app_path, "resources")
        self.output_path: AnyStr = join(self.app_path, "output")
        self.logs_path: AnyStr = join(self.output_path, "logs")
        self.tensorboard_path: AnyStr = join(self.logs_path, self.run)
        self.model_path: AnyStr = join(self.output_path, "models")
        self.student_path: AnyStr = join(self.model_path, "student")
        self.output_files: AnyStr = join(self.output_path, "output_files")
        self.retrieved_documents_folder: AnyStr = join(
            self.output_path, "retrieved_documents")
        self.dataset_folder: AnyStr = join(self.resources_path, "data")
        self.annotations_folder: AnyStr = join(
            self.resources_path, "annotations")
        self.results_folder: AnyStr = join(self.resources_path, "results")
        self._create_output_dirs()

    def _create_output_dirs(self):
        """
        Create missing directories (e.g. logs, output etc)
        """
        makedirs(self.output_path, exist_ok=True)
        makedirs(self.logs_path, exist_ok=True)
        makedirs(self.tensorboard_path, exist_ok=True)
        makedirs(self.model_path, exist_ok=True)
        makedirs(self.student_path, exist_ok=True)
        makedirs(self.output_files, exist_ok=True)
        makedirs(self.dataset_folder, exist_ok=True)
        makedirs(join(self.dataset_folder, "adu"), exist_ok=True)
        makedirs(join(self.dataset_folder, "rel"), exist_ok=True)
        makedirs(join(self.dataset_folder, "stance"), exist_ok=True)
        makedirs(self.results_folder, exist_ok=True)
        makedirs(self.retrieved_documents_folder, exist_ok=True)

    def _get_base_path(self, base_name: AnyStr) -> AnyStr:
        """
        Create the base full path to the directory where each model will be saved

        Args
            base_name (str): the name of the model

        Returns
            str: the path to the directory of the model
        """
        # Create a base path:
        # if base_name == "adu":
        #     properties = self.properties["seq_model"]
        # else:
        #     properties = self.properties["class_model"]
        # bert_kind = properties["bert_kind"][base_name].replace("/", "-")
        # embedding_names = f"bert-{bert_kind}"
        # layers = properties["rnn_layers"] if base_name == "adu" else properties["layers"]
        # base_path = f"{base_name}-" + '-'.join([
        #     str(embedding_names),
        #     'hs=' + str(properties["hidden_size"]),
        #     'hl=' + str(layers),
        #     'crf=' + str(properties["use_crf"]),
        #     "optmizer=" + properties["optimizer"],
        #     'lr=' + str(properties["learning_rate"]),
        #     'bs=' + str(properties["mini_batch_size"])
        # ])
        base_path = join(self.model_path, base_name)
        try:
            os.makedirs(base_path)
        except (OSError, Exception):
            pass
        return base_path

    def _config_elastic_search(self):
        self.elastic_retrieve: ElasticSearchConfig = ElasticSearchConfig(properties=self.properties,
                                                                         properties_file=self.properties_file,
                                                                         resources_folder=self.resources_path,
                                                                         elasticsearch="retrieve",
                                                                         logger=self.app_logger)
        self.elastic_save: ElasticSearchConfig = ElasticSearchConfig(properties=self.properties,
                                                                     properties_file=self.properties_file,
                                                                     resources_folder=self.resources_path,
                                                                     elasticsearch="save", logger=self.app_logger)


class Notification:

    def __init__(self, app_config: AppConfig):
        self.app_config: AppConfig = app_config
        self.app_logger: logging.Logger = app_config.app_logger
        self.properties: Dict = app_config.properties
        self.resources_path: AnyStr = app_config.resources_path
        self.properties_file: AnyStr = app_config.properties_file
        self.logs_path: AnyStr = app_config.logs_path
        self.log_filename: AnyStr = app_config.log_filename
        self._config_email(config=self.properties["config"])

    def _config_email(self, config):
        """
        Email configuration in order to get notification when the program has finished

        Args
            config (dict): configuration parameters for email
        """
        config_email = config.get("email", None)
        self.do_send_email = False
        if config_email and type(config_email) == dict:
            self.sender_email = config_email["sender"]
            self.receiver_email = config_email["receiver"]
            self.password = config_email["password"]
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(self.sender_email, self.password)
                self.do_send_email = True
            except (BaseException, Exception):
                self.app_logger.error(
                    "Connecting to email account failed! Check your credentials!")
                self.do_send_email = False

    def notify_ics(self, ids_list: List[AnyStr], kind: AnyStr = "arg_mining"):
        """
        Function to notify ICS API for any updates in the Elasticsearch. Uses different API endpoints based on the
        kind parameter. The possible notifications are: argument mining updates for new documents, clustering updates
        for cross-document relations

        Args
            | ids_list (list): list of ids in the Elasticsearch (documents or relations)
            | kind (str): the kind of update, possible values --> arg_mining, clustering
        """
        routing_key = "dlabqueue" if kind == "arg_mining" else "dlab-cross-docs"
        self.app_logger.info(f"Notification to queue: {routing_key}")
        properties = self.properties["eval"]["notify"]
        url = properties["url"]
        username = properties["username"]
        password = properties["password"]
        data = {"properties": {"delivery_mode": 2}, "routing_key": routing_key, "payload": json.dumps(ids_list),
                "payload_encoding": "string"}
        creds = f"{username}:{password}"
        creds_bytes = creds.encode("ascii")
        base64_bytes = base64.b64encode(creds_bytes)
        base64_msg = base64_bytes.decode("ascii")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Basic {base64_msg}"}
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                self.app_logger.info("Request to ICS was successful!")
            else:
                self.app_logger.error(
                    f"Request to ICS failed with status code: {response.status_code} and message:{response.text}")
        except(BaseException, Exception) as e:
            self.app_logger.error(f"Request to ICS failed: {e}")

    def send_email(self, body: AnyStr, subject: AnyStr = None):
        """
        Function to send a notification email upon completion of the program

        Args
            body (str): the body message
            subject (str): the subject of the email
        """
        if not self.do_send_email:
            return
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

    def __init__(self, logger, properties: Dict, properties_file: AnyStr, resources_folder: AnyStr,
                 elasticsearch: AnyStr):
        """
        Constructor of the ElasticSearchConfig class

        Args
            | properties (dict): a dictionary with the configuration parameters
            | elasticsearch (str): valid values are save or retrieve in order to determine which parameters to use
            The elastic_save parameters are configurations for the debatelab elasticsearch while the elastic_retrieve
            properties are the configurations, credentials etc of the socialobservatory elasticsearch.
        """
        self.logger = logger
        self.resources_folder: AnyStr = resources_folder
        self.properties_file: AnyStr = properties_file
        self.properties: Dict = properties
        config_prop = properties["config"]
        elastic_prop = config_prop["elastic_save"] if elasticsearch == "save" else config_prop["elastic_retrieve"]
        self.username: AnyStr = elastic_prop["username"]
        self.password: AnyStr = elastic_prop["password"]
        self.host: AnyStr = elastic_prop["host"]
        self.port: int = elastic_prop["port"]
        self.ssh_port: int = elastic_prop["ssh"]["port"]
        self.ssh_username: AnyStr = elastic_prop["ssh"]["username"]
        self.ssh_password: AnyStr = elastic_prop["ssh"]["password"]
        self.ssh_key: AnyStr = elastic_prop["ssh"]["key_path"]
        self.connect: AnyStr = elastic_prop["connect"]

        try:
            self._init_ssh_tunnel()
            self._init_elasticsearch_client()
            self.connected = True
            self.logger.info(
                f"Connected to ssh client to {elasticsearch} documents")
        except (BaseException, Exception) as e:
            self.logger.error(
                f"An error occurred while trying to connect via ssh: {e}")
            try:
                self.logger.warning(
                    "Could not connect via ssh. Trying via http...")
                self._init_elastic_search_client_http()
                self.connected = True
                self.logger.info(
                    f"Connected to elasticsearch via http to {elasticsearch} documents")
            except(BaseException, Exception) as e:
                self.logger.warning(
                    "Could not connect to ElasticSearch with ssh or http")
                self.logger.error(e)
                self.connected = False

    def _init_elasticsearch_client(self, timeout=120):
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

    def _init_elastic_search_client_http(self, timeout=120):
        self.elasticsearch_client = Elasticsearch([{
            'host': self.host,
            'port': self.port,
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
        if self.connected:
            del self.elasticsearch_client
            [t.close() for t in threading.enumerate()
             if t.__class__.__name__ == "Transport"]
            self.tunnel.stop()

    def truncate_elasticsearch(self, index):
        """
        Delete all entries in the elasticsearch
        """
        Search(using=self.elasticsearch_client,
               index=index).query("match_all").delete()

    def retrieve_documents(self, previous_date=None, retrieve_kind="file", output_folder="app/resources/retrieved_documents"):
        search_id = None
        expected_num_results = None
        if retrieve_kind == "explicit":
            field = self.properties["eval"]["explicit_field"]
            file_path = join(self.resources_folder,
                             self.properties["eval"]["file_path"])
            # read the list of urls from the file:
            with open(file_path, "r") as f:
                if field in ("links", "_id"):
                    value_list = [line.rstrip() for line in f]
            self.logger.info(
                f"Fetching documents from explicit list of {len(value_list)} {field} items")
            match_arg = {field: value_list}
            search_articles = Search(
                using=self.elasticsearch_client, index='articles').filter('terms', **match_arg)
            expected_num_results = len(value_list)
            search_id = f"{retrieve_kind}_{field}_{len(value_list)}"
        else:
            delta = self.properties["eval"]["delta_days"]
            today_date = datetime_base.date.today()
            previous_date = today_date - datetime_base.timedelta(days=delta)
            search_term = self.properties["eval"]["search_term"]
            date_range = {'gt': previous_date, 'lte': today_date}
            search_id = f"{str(today_date)}_range_{','.join([k + '_' + str(v) for (k,v) in date_range.items()])}"
            self.logger.info(f"Fetching documents from search: {search_id}")

            date_range = {'gt': str(previous_date), 'lte': str(today_date)}
            if search_term:
                try:
                    against = self.properties["eval"]["search_against"]
                except KeyError:
                    against = "title"
                if type(search_term) is not list:
                    search_term = [search_term]
                queries = []
                for st in search_term:
                    mtch = {against: st}
                    queries.append(Q('match', **mtch))
                # if len(queries) == 1:
                #     queries = queries[0]
                query = Q('bool', must=queries, filter=[
                    {"range": {"date": date_range}}])
                search_articles = Search(
                    using=self.elasticsearch_client, index='articles')
                search_articles.query = query
                search_id += "_terms_" + ",".join(search_term)
            else:
                search_articles = Search(using=self.elasticsearch_client, index='articles').filter(
                    'range', date=date_range)
        documents = []
        for hit in search_articles.scan():
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            if not document["content"].startswith(document["title"]):
                document["content"] = document["title"] + \
                    "\r\n\r\n" + document["content"]
            self.logger.debug(
                f"Appending doc from {document['date']} titled: {document['title']}")
            documents.append(document)
        # self.update_last_retrieve_date()
        self.logger.debug(f"Got {len(documents)} docs")

        try:
            if self.properties["prep"]["newline_norm"] is True:
                self.logger.debug(f"Normalizing newlines")
                for doc in documents:
                    doc['content'] = normalize_newlines(doc['content'])
        except KeyError:
            pass
        if expected_num_results is not None:
            if len(documents) != expected_num_results:
                raise ValueError(
                    f"Expected {expected_num_results} docs but got {len(documents)}")

        # save docs
        with open(join(output_folder, f"retrieved_documents_{len(documents)}_{search_id}.json"), "w") as f:
            self.logger.info(f"Saving obtained docs to {f.name}")
            json.dump(documents, f)
        return documents

    def update_last_retrieve_date(self):
        path_to_properties = join(self.resources_folder, self.properties_file)
        # update last search date
        if self.properties["eval"]["retrieve"] == "date":
            self.properties["eval"]["last_date"] = datetime.now()
            with open(path_to_properties, "w") as f:
                yaml.dump(self.properties, f)

    def save_document(self, document):
        self.elasticsearch_client.index(index='debatelab', ignore=400, refresh=True,
                                        doc_type='docket', id=document["id"],
                                        body=document)

    def save_relation(self, relation):
        self.elasticsearch_client.index(index='debatelab-crossdoc-relations', ignore=400, refresh=True,
                                        doc_type='docket', id=relation["id"],
                                        body=relation)
