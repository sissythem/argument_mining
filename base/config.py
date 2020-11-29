import logging
import smtplib
import ssl
import uuid
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import mkdir
from os.path import exists, join

import yaml


class AppConfig:
    properties_file = "properties.yaml"
    example_properties = "example_properties.yaml"
    documents_pickle = "documents.pickle"
    pickle_sentences_filename = "sentences.pickle"
    pickle_segments_filename = "segments.pickle"
    relations_pickle_file = "relations.pickle"
    stances_pickle_file = "stances.pickle"
    pickle_relations_labels = "relations_labels.pickle"
    log_filename = 'logs_%s' % datetime.now().strftime('%Y%m%d-%H%M%S')

    def __init__(self, app_path):
        # paths & directories
        self.app_path = app_path
        self.run = uuid.uuid4().hex
        self._create_paths()
        self._create_output_dirs()
        self.properties = {}
        self.data_file = ""
        self.app_logger = None

    def configure(self):
        # logging
        self.app_logger = self._config_logger()
        self.app_logger.info("Run id: {}".format(self.run))

        # properties
        self.properties = self._load_properties()
        config = self.properties["config"]
        self.data_file = config["data"]["filename"]
        self._config_email(config=config)

    def _create_paths(self):
        self.resources_path = join(self.app_path, "resources")
        self.output_path = join(self.app_path, "output")
        self.logs_path = join(self.output_path, "logs")
        self.model_path = join(self.output_path, "model")
        self.tensorboard_path = join(self.output_path, "runs")
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
