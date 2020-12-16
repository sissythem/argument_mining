import json
import traceback
from os.path import join

from elasticsearch_dsl import Search
from ellogon import esclient_swo
import requests, json, os
from elasticsearch import Elasticsearch
import utils
from arg_mining import AduModel, RelationsModel, ArgumentMining
from training_data import DataLoader
from utils import AppConfig


def preprocess(app_config):
    logger = app_config.app_logger
    data_loader = DataLoader(app_config=app_config)
    logger.info("Creating CSV file in CONLL format for ADUs classification")
    data_loader.load_adus()
    logger.info("Creating CSV file in CONLL format for relations classification")
    data_loader.load_relations()
    logger.info("Creating CSV file in CONLL format for stance classification")


def train(app_config):
    logger = app_config.app_logger
    logger.info("Training ADU classifier")
    adu_model = AduModel(app_config=app_config)
    adu_model.train()
    logger.info("ADU Training is finished!")
    logger.info("Training relations model")
    rel_model = RelationsModel(app_config=app_config, dev_csv=app_config.rel_dev_csv,
                               train_csv=app_config.rel_train_csv, test_csv=app_config.rel_test_csv,
                               base_path=app_config.rel_base_path, model_name="rel")
    rel_model.train()
    logger.info("Relations training finished!")
    logger.info("Training stance model")
    stance_model = RelationsModel(app_config=app_config, dev_csv=app_config.stance_dev_csv, model_name="stance",
                                  train_csv=app_config.stance_train_csv, test_csv=app_config.stance_test_csv,
                                  base_path=app_config.stance_base_path)
    stance_model.train()
    logger.info("Stance training finished!")


def evaluate(app_config):
    logger = app_config.app_logger
    eval_source = app_config.properties["eval"]["source"]
    if eval_source == "elasticsearch":
        eval_from_elasticsearch(app_config)
    else:
        eval_from_file(app_config=app_config)
    logger.info("Evaluation is finished!")


def eval_from_elasticsearch(app_config):
    logger = app_config.app_logger
    client = esclient_swo.elastic_server_client
    file_path = join(app_config.resources_path, "kasteli_34_urls.txt")
    # read the list of urls from the file:
    with open(file_path, "r") as f:
        urls = [line.rstrip() for line in f]
    search_articles = Search(using=client, index='articles').filter('terms', link=urls)
    # print(search_articles.to_dict())
    found = 0
    arg_mining = ArgumentMining(app_config=app_config)
    for hit in search_articles.scan():
        document = hit.to_dict()
        if not document["content"].startswith(document["title"]):
            document["content"] = document["title"] + "\r\n\r\n" + document["content"]
        arg_mining.predict(document=document)
        found += 1
    logger.info(f"Found documents: {found}")
    save_output_files_to_elasticsearch(path=app_config.out_files_path)


def save_output_files_to_elasticsearch(path):
    es = Elasticsearch([{
        'host': '143.233.226.60',
        'port': "9200",
        'http_auth': ('debatelab', 'SRr4TqV9rPjfzxUmYcjR4R92')
    }], timeout=60)
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            f = open(filename)
            docket_content = f.read()
            # Send the data into es
            es.index(index='debatelab', ignore=400, doc_type='docket',
                     body=json.loads(docket_content))


def eval_from_file(app_config, filename="kasteli.json"):
    logger = app_config.app_logger
    logger.info("Evaluating using file: {}".format(filename))
    file_path = join(app_config.resources_path, filename)
    with open(file_path, "r") as f:
        data = json.load(f)
    documents = data["data"]["documents"]
    if documents:
        arg_mining = ArgumentMining(app_config=app_config)
        for document in documents:
            document = utils.get_initial_json(document["name"], document["text"])
            arg_mining.predict(document=document)


def main():
    app_config: AppConfig = AppConfig()
    try:
        properties = app_config.properties
        tasks = properties["tasks"]
        if "prep" in tasks:
            preprocess(app_config=app_config)
        if "train" in tasks:
            train(app_config=app_config)
        if "eval" in properties["tasks"]:
            evaluate(app_config=app_config)
        app_config.send_email(body="Argument mining pipeline finished successfully",
                              subject="Argument mining run: {}".format(app_config.run))
    except(BaseException, Exception):
        app_config.app_logger.error(traceback.format_exc())
        app_config.send_email(
            body="Argument mining pipeline finished with errors".format(traceback.format_exc(limit=100)),
            subject="Error in argument mining run: {}".format(app_config.run))
    finally:
        try:
            esclient_swo.stop()
        except(BaseException, Exception):
            pass


if __name__ == '__main__':
    main()
