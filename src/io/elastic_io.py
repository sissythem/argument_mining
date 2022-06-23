import yaml
import threading
import datetime as datetime_base
from datetime import datetime
from os.path import join

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from sshtunnel import SSHTunnelForwarder

from typing import Dict, AnyStr
import logging
import copy


def convert_doc_to_save_format(document):
    # finalize by only retaining required keys
    doc = copy.deepcopy(document)
    keys = ["type", "starts", "ends", "segment", "tokens", "confidence", "id"]
    adus = doc['annotations']['ADUs']
    for i, adu in enumerate(adus):
        reformatted_adu = {k: adu[k] for k in keys}
        if "stance" in adu:
            reformatted_adu["stance"] = adu["stance"]
        doc['annotations']['ADUs'][i] = reformatted_adu
    return doc


class ElasticSearchConfig:
    """
    Class to configure an Elasticsearch Client
    """

    def __init__(self, config: Dict):
        """
        Constructor of the ElasticSearchConfig class

        Args
            | properties (dict): a dictionary with the configuration parameters
            | elasticsearch (str): valid values are save or retrieve in order to determine which parameters to use
            The elastic_save parameters are configurations for the debatelab elasticsearch while the elastic_retrieve
            properties are the configurations, credentials etc of the socialobservatory elasticsearch.
        """
        self.search_ids = []

        self.username: AnyStr = config["https"]["username"]
        self.password: AnyStr = config["https"]["password"]
        self.host: AnyStr = config["https"]["host"]
        self.port: int = config["https"]["port"]
        self.ssh_port: int = config["ssh"]["port"]
        self.ssh_username: AnyStr = config["ssh"]["username"]
        self.ssh_password: AnyStr = config["ssh"]["password"]
        self.ssh_key: AnyStr = config["ssh"]["key_path"]
        self.connect: AnyStr = config["connect"]

        try:
            self._init_ssh_tunnel()
            self._init_elasticsearch_client()
            self.connected = True
            logging.info(
                f"Connected to ssh client")
        except (BaseException, Exception) as e:
            logging.error(
                f"An error occurred while trying to connect via ssh: {e}")
            try:
                logging.warning(
                    "Could not connect via ssh. Trying via http...")
                self._init_elastic_search_client_http()
                self.connected = True
                logging.info(
                    f"Connected to elasticsearch via http")
            except(BaseException, Exception) as e:
                logging.warning(
                    "Could not connect to ElasticSearch with ssh or http")
                logging.error(e)
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
            try:
                self.tunnel.stop()
            except AttributeError:
                pass
            self.connected = False

    def retrieve_documents(self, **kwargs):
        exclude = {}

        mode = kwargs['mode']
        mode_settings = kwargs[mode]
        expected_num_results = None
        max_num_results = None
        if mode in ("data", "disk"):
            rtype = mode_settings['type']

            if mode == "disk":
                path = mode_settings['path']
                # read the list of urls from the file:
                with open(path, "r") as f:
                    if rtype in ("link", "_id"):
                        value_list = [line.rstrip() for line in f]
                    else:
                        raise ValueError(f"Undefined field for explicit document retrieval: {rtype}")
                logging.info(
                    f"Fetching documents from explicit list of {len(value_list)} {rtype} items")

            elif mode == "data":
                value_list = mode_settings['values']

            match_arg = {rtype: value_list}
            search_articles = Search(
                using=self.elasticsearch_client, index='articles').filter('terms', **match_arg)
            expected_num_results = len(value_list)
            search_id = f"{mode}_{rtype}_{len(value_list)}"
        else:
            max_num_results = mode_settings.get("max_num_results", None)
            delta = mode_settings["delta_days"]
            try:
                search_term = mode_settings["terms"]
            except KeyError:
                search_term = ""

            today_date = datetime_base.datetime.today()
            previous_date = today_date - datetime_base.timedelta(days=delta)
            date_range = {'gt': previous_date, 'lte': today_date}
            search_id = f"{str(today_date)}_range_{','.join([k + '_' + str(v) for (k, v) in date_range.items()])}"
            logging.info(f"Fetching documents from search: {search_id}")

            # to unix millis
            previous_millis = round(previous_date.timestamp() * 1000)
            today_millis = round(today_date.timestamp() * 1000)

            date_range = {'gt': str(previous_millis), 'lte': str(today_millis)}
            if search_term:
                try:
                    search_field = mode_settings["field"]
                except KeyError:
                    search_field = "title"

                try:
                    exclude = mode_settings["exclude"]
                    for k, v in exclude.items():
                        if type(v) is not list:
                            with open(v) as f:
                                v = [x.strip() for x in f.readlines()]
                            exclude[k] = v
                except KeyError:
                    pass
                if type(search_term) is not list:
                    search_term = [search_term]
                queries = []
                for st in search_term:
                    mtch = {search_field: st}
                    queries.append(Q('match', **mtch))
                # if len(queries) == 1:
                #     queries = queries[0]
                query = Q('bool', must=queries, filter=[
                    {"range": {"date": date_range}}])
                search_articles = Search(
                    using=self.elasticsearch_client, index='articles')
                search_articles.query = query
                search_id += "_field_" + search_field + "_terms_" + ",".join(search_term)
            else:
                search_articles = Search(using=self.elasticsearch_client, index='articles').filter(
                    'range', date=date_range)
        documents = []
        # scanning yields "Trying to create too many scroll contexts" error
        # it = search_articles.scan()
        search_articles = search_articles[0:search_articles.count()]
        # it = search_articles.extra(track_total_hits=True).execute()
        it = search_articles.execute()
        for i, hit in enumerate(it):
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            if not document["content"].startswith(document["title"]):
                document["content"] = document["title"] + \
                                      "\r\n\r\n" + document["content"]
            logging.debug(
                f"Appending doc from {document['date']} titled: {document['title']}")
            if exclude:
                for k, v in exclude.items():
                    doc_val = document[k]
                    if doc_val in v:
                        logging.info(f"Skipping doc titled: {document['title']} from {k} excluded value: {doc_val}")
                        continue
            try:
                if mode_settings["interactive"]:
                    print("#########################")
                    print(f"Doc {i + 1}/{len(it)}, ({len(documents)} stored):")
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
            except KeyError:
                documents.append(document)
            if max_num_results and len(documents) >= max_num_results:
                logging.info(f"Stopping document retrieval due to a limit of {max_num_results} docs specified")
                break

        # self.update_last_retrieve_date()
        logging.info(f"Retrieved {len(documents)} docs from ES")
        if value_list:
            not_found = [l for l in value_list if l not in [d['link'] for d in documents]]
            if not_found:
                logging.info(f"Could not find {len(not_found)} docs:")
                for nf in not_found:
                    logging.info(str(nf))

        if expected_num_results is not None:
            try:
                err = mode_settings["error_on_missing"]
            except KeyError:
                err = False
            if len(documents) != expected_num_results and err:
                raise ValueError(
                    f"Expected {expected_num_results} docs but got {len(documents)}")

        self.search_ids.append(search_id)
        return documents

    def get_last_search_id(self):
        return self.search_ids[-1]

    def update_last_retrieve_date(self):
        path_to_properties = join(self.resources_folder, self.properties_file)
        # update last search date
        if self.properties["eval"]["retrieve"] == "date":
            self.properties["eval"]["last_date"] = datetime.now()
            with open(path_to_properties, "w") as f:
                yaml.dump(self.properties, f)

    def save_document(self, document):
        processed_document = convert_doc_to_save_format(document)
        self.elasticsearch_client.index(index='debatelab', ignore=400, refresh=True,
                                        doc_type='docket', id=document["id"],
                                        body=processed_document)

    def save_relation(self, relation):
        self.elasticsearch_client.index(index='debatelab-crossdoc-relations', ignore=400, refresh=True,
                                        doc_type='docket', id=relation["id"],
                                        body=relation)

    def get_all_documents(self):
        return self._get_all('debatelab')

    def get_all_relations(self):
        return self._get_all('debatelab-crossdoc-relations')

    def _get_all(self, index):
        iter = Search(using=self.elasticsearch_client, index=index).scan()
        res = []
        for i, hit in enumerate(iter):
            if i > 0 and i % 100 == 0:
                logging.info(f"Retrieved {i} elements from [{index}]...")
            document = hit.to_dict()
            document["id"] = hit.meta["id"]
            res.append(document)
        return res

    def delete_all_relations(self):
        self._delete_all("debatelab-crossdoc-relations")

    def delete_all_documents(self):
        self._delete_all("debatelab")

    def _delete_all(self, index):
        self.elasticsearch_client.delete_by_query(index=[index], body={"query": {"match_all": {}}})

    def delete_document(self, document):
        self.elasticsearch_client.delete(index='debatelab', refresh=True, doc_type='docket', id=document["id"])

    def delete_relation(self, relation):
        self.elasticsearch_client.delete(index='debatelab-crossdoc-relations', refresh=True, doc_type='docket',
                                         id=relation["id"])
