import yaml
import threading
import datetime as datetime_base
from datetime import datetime
from os.path import exists, join
from os import makedirs

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from sshtunnel import SSHTunnelForwarder

from src.utils.utils import normalize_newlines
from typing import List, Dict, AnyStr
import json


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

    def retrieve_documents(self, output_folder="resources/retrieved_documents"):
        exclude = {}
        retrieval_mode = self.properties["eval"]["retrieve"]
        expected_num_results = None
        if retrieval_mode == "explicit":
            field = self.properties["eval"]["explicit_field"]
            file_path = join(self.resources_folder,
                             self.properties["eval"]["file_path"])
            # read the list of urls from the file:
            with open(file_path, "r") as f:
                if field in ("link", "_id"):
                    value_list = [line.rstrip() for line in f]
                else:
                    raise ValueError(f"Undefined field for explicit document retrieval: {field}")
            self.logger.info(
                f"Fetching documents from explicit list of {len(value_list)} {field} items")
            match_arg = {field: value_list}
            search_articles = Search(
                using=self.elasticsearch_client, index='articles').filter('terms', **match_arg)
            expected_num_results = len(value_list)
            search_id = f"{retrieval_mode}_{field}_{len(value_list)}"
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
                if "exclude_list" in self.properties["eval"]:
                    exclude = self.properties["eval"]["exclude_list"]
                    for k, v in exclude.items():
                        if type(v) is not list:
                            with open(v) as f:
                                v = [x.strip() for x in f.readlines()]
                            exclude[k] = v

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
                search_id += "_field_" + against + "_terms_" + ",".join(search_term)
            else:
                search_articles = Search(using=self.elasticsearch_client, index='articles').filter(
                    'range', date=date_range)
        documents = []
        it = list(search_articles.scan())
        for i, hit in enumerate(it):
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            if not document["content"].startswith(document["title"]):
                document["content"] = document["title"] + \
                    "\r\n\r\n" + document["content"]
            self.logger.debug(
                f"Appending doc from {document['date']} titled: {document['title']}")
            if exclude:
                for k, v in exclude.items():
                    doc_val = document[k]
                    if doc_val in v:
                        self.logger.info(f"Skipping doc titled: {document['title']} from {k} excluded value: {doc_val}")
                        continue
            if self.properties["eval"]["interactive"]:
                print("#########################")
                print(f"Doc {i+1}/{len(it)}, ({len(documents)} stored):")
                search_id = "interactive_" + search_id
                print()
                print("----------------------------")
                print(f"Title: [{document['title']}]")
                print("----------------------------")
                print("Content:\n==============\n", document["content"])
                print("----------------------------")
                yn = input("Keep?")
                print("----------------------------")
                if "y" == yn.lower():
                    documents.append(document)
                if "e" == yn.lower():
                    break
            else:
                documents.append(document)
        # self.update_last_retrieve_date()
        self.logger.debug(f"Got {len(documents)} docs")

        try:
            if self.properties["prep"]["newline_norm"]:
                self.logger.info(f"Normalizing newlines")
                for doc in documents:
                    doc['content'] = normalize_newlines(doc['content'])
        except KeyError:
            pass
        if expected_num_results is not None:
            if len(documents) != expected_num_results:
                raise ValueError(
                    f"Expected {expected_num_results} docs but got {len(documents)}")

        # save docs
        makedirs(output_folder, exist_ok=True)
        with open(join(output_folder, f"retrieved_documents_{len(documents)}_{search_id}.json"), "w") as f:
            self.logger.info(f"Saving obtained docs to {f.name}")
            json.dump(documents, f, ensure_ascii=False)
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
