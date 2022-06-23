from os.path import join
import logging
import requests
import smtplib
import base64
import ssl
import json
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders


class Notification:

    def __init__(self, config):
        self.url = config["url"]
        self.username = config["username"]
        self.password = config["password"]

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
                logging.error(
                    "Connecting to email account failed! Check your credentials!")
                self.do_send_email = False

    def notify_ics(self, ids_list, kind="arg_mining"):
        """
        Function to notify ICS API for any updates in the Elasticsearch. Uses different API endpoints based on the
        kind parameter. The possible notifications are: argument mining updates for new documents, clustering updates
        for cross-document relations

        Args
            | ids_list (list): list of ids in the Elasticsearch (documents or relations)
            | kind (str): the kind of update, possible values --> arg_mining, clustering
        """
        routing_key = "dlabqueue" if kind == "arg_mining" else "dlab-cross-docs"
        logging.info(f"Notification to queue: {routing_key}")

        data = {"properties": {"delivery_mode": 2}, "routing_key": routing_key, "payload": json.dumps(ids_list),
                "payload_encoding": "string"}
        creds = f"{self.username}:{self.password}"
        creds_bytes = creds.encode("ascii")
        base64_bytes = base64.b64encode(creds_bytes)
        base64_msg = base64_bytes.decode("ascii")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Basic {base64_msg}"}
        try:
            response = requests.post(self.url, json=data, headers=headers)
            if response.status_code == 200:
                logging.info("Request to ICS was successful!")
            else:
                logging.error(
                    f"Request to ICS failed with status code: {response.status_code} and message:{response.text}")
        except(BaseException, Exception) as e:
            logging.error(f"Request to ICS failed: {e}")

    def send_email(self, body, subject=None):
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


