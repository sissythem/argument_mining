import json
import re
import threading
from datetime import datetime
from os import mkdir, getcwd
from os.path import join, exists
from string import punctuation

from elasticsearch import Elasticsearch

# from elasticsearch_dsl import Search, Q
from ellogon import tokeniser
from genson import SchemaBuilder
from sshtunnel import SSHTunnelForwarder

from src.pipeline.validation import JsonValidator
from src.utils.config import AppConfig


def tokenize(text, punct=True):
    return list(tokeniser.tokenise_no_punc(text)) if not punct else list(tokeniser.tokenise(text))


def get_punctuation_symbols():
    """
    Function to get a set with punctuation symbols

    Returns
        | set: a set with all the punctuation symbols
    """
    punc = list(set(punctuation))
    punc += ["´", "«" "»"]
    punc = set(punc)
    return punc


def join_sentences(tokenized_sentences):
    """
    Function to create a correct string (punctuation in the correct position - correct spaces)

    Args
        | tokenized_sentences (list): a list of sentences. Each sentence is a tuple with the respective tokens

    Returns
        | list: a list of strings (i.e. the sentences)
    """
    sentences = []
    punc = get_punctuation_symbols()
    for sentence in tokenized_sentences:
        sentence = "".join(w if set(w) <= punc else f" {w}" for w in sentence).lstrip()
        sentence = sentence.replace("( ", " (")
        sentence = sentence.replace("« ", " «")
        sentence = sentence.replace(" »", "» ")
        sentence = sentence.replace('" ', ' "')
        sentence = sentence.replace("\n", " ")
        sentence = re.sub(" +", " ", sentence)
        sentences.append(sentence)
    return sentences


def stop(tunnel, elastic_server_client):
    del elastic_server_client
    [t.close() for t in threading.enumerate() if t.__class__.__name__ == "Transport"]
    tunnel.stop()


def start():
    tunnel = SSHTunnelForwarder(
        ssh_address="143.233.226.60",
        ssh_port=222,
        ssh_username="debatelab",
        ssh_private_key="/home/sthemeli/.ssh/id_rsa",
        remote_bind_address=('127.0.0.1', 9200),
        compression=True
    )
    tunnel.start()
    return tunnel


def elastic_ssh():
    tunnel = start()
    return Elasticsearch([{
        'host': 'localhost',
        'port': tunnel.local_bind_port,
        'http_auth': ("debatelab", "SRr4TqV9rPjfzxUmYcjR4R92")
    }], timeout=60), tunnel


def elastic_http():
    return Elasticsearch([{
        'host': "143.233.226.60",
        'port': 9200,
        'http_auth': ("debatelab", "SRr4TqV9rPjfzxUmYcjR4R92")
    }], timeout=60)


def get_articles(elastic_client, ids):
    res = elastic_client.mget(index="debatelab", body={"ids": ids})
    return res["docs"] if res else None


def export_schema(data):
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    for doc in data:
        builder.add_object(doc)
    print(builder.to_schema())
    print(builder.to_json(indent=2))


def export_to_json_files(data, do_validation=False, with_sentences=False):
    if not exists(join(getcwd(), "jsons")):
        mkdir(join(getcwd(), "jsons"))
    validator = get_validator() if do_validation else None
    for document in data:
        document = document["_source"]
        if with_sentences:
            sentences = join_sentences(tokenized_sentences=tokenize(text=document["content"]))
            document["sentences"] = sentences
        if do_validation:
            validate(validator=validator, document=document)
        with open(join(getcwd(), "jsons", f"{document['id']}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(document, ensure_ascii=False, indent=4))


def validate(validator, document):
    validation_errors = validator.validate(document=document)
    print(validation_errors)
    if validation_errors:
        timestamp = datetime.now()
        filename = f"{document['id']}_{timestamp}.json"
        file_path = join(getcwd(), f"output/output_files/{filename}")
        txt_file_path = file_path.replace(".json", ".txt")
        with open(file_path, "w", encoding='utf8') as f:
            f.write(json.dumps(document, indent=4, sort_keys=False, ensure_ascii=False))
        with open(txt_file_path, "w") as f:
            for validation_error in validation_errors:
                f.write(validation_error.value + "\n")


def get_validator():
    app_config = AppConfig()
    return JsonValidator(app_config=app_config)


def main(connection_type="http", export=("articles", "schema"), do_validation=False, with_sentences=True):
    swo_ids = ['baadf1a74546e9e1bc377b1c4304d67428003895', '614cfba661d2e321836e5e84f30bf345184820b6',
               'fd886749a33afb6c0cc624d9d8d7d838c2157f8a', '12cf72439683786985202df62368d6e52947c3ae',
               'dc3c5604355331ef65b067207d809f6dc4393cda', 'da556dfd49831f9feb76bda7a0867dbf1f2e6990',
               '1c0a87dff8d2734b9e9b0ea136505d84866a494c', '0a7e4aa11918ed0e3d294f4939c804244571863b',
               '4d217607564f2f7a8f00c081a82c5e5f4348c567', '1a449b3a8fa855247e31c61127bee16fe2100049',
               'a14334b2b4403346421df04747b0e9f165589d5b', '2a0ab8adec1f2fc1ed1f61e27cf016476742128a',
               '4e3b0d0edc45d2749150d26b216527e26ccde839', '861b20c562b199d37a494777f14d3310ef0e90b0',
               '65bb74d6d2397e6ed670dacb0e63a29134df0569', '1d7bc5152a97a64d6c4c3f83950610847153d0ac',
               '9069725772edfa6ae602b15f367a0ada4cc195f6', '6f82414348ff36925f64c27e96fb38af75627949',
               '9440ef6c80c8d7077c7b70d6a2f11c6c4537e0d3', 'd09c96db7afbb6b7a1b92c16b9fbb2f8705d9ee3',
               'b6406b7750094ec693d1b642607a450793ef5d5c', '97b898806a620d8605357c8161aae2ebb736e3d3',
               '8c00e6757924638c2f8c06408c7040b98316aaa1', '0f69cb23b2d25ca7444840ea55e642f6dba07e9b',
               '2ba337ba6df1f7ef8e1ee85d52603e53c22fc1cd', '30a0e34bad2fd509340d021bfa609f6a9f6388d7',
               '45d4084b99f8beee7d13fb21dcfd93b49cfb2778', '97e0f737ba0f3a5b60e32630b90e0b242ac2a960',
               '5a6a4222c17b66a4f8ecd0750762d316f666db0a', '2a6a5690affc0a5733ef5fc4354a4d32aa91e230',
               'aed988e95568cf6e999ef4ec852d37c9482e7801', 'dca77f44643e63f2a49caf3455c94ad809e18ca0',
               'e6d74b3832e335b73961645fa42667c106508782']

    if connection_type == "http":
        elastic_client = elastic_http()
        tunnel = None
    else:
        elastic_client, tunnel = elastic_ssh()
    articles = get_articles(elastic_client=elastic_client, ids=swo_ids)
    if "articles" in export:
        export_to_json_files(data=articles, with_sentences=with_sentences, do_validation=do_validation)
    if "schema" in export:
        export_schema(data=articles)
    if connection_type == "ssh":
        stop(tunnel=tunnel, elastic_server_client=elastic_client)


if __name__ == '__main__':
    main(connection_type="http", export=("articles"), do_validation=False, with_sentences=True)
