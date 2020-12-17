import json
from os import getcwd, listdir
from os.path import join

output_path = join(getcwd(), "output")
backup_path = join(output_path, "output_files_backup")
elastic_path = join(output_path, "output_files")


def read_docs(path):
    doc_dict = {}
    files = listdir(path)
    if files:
        for filename in files:
            file_path = join(path, filename)
            with open(file_path, "r") as f:
                document = json.load(f)
                title = document["title"][:int(len(document["title"]) / 2)]
                doc_dict[title] = (document["title"], document["content"])
    return doc_dict


def main():
    elasticsearch_docs, backup_docs = read_docs(elastic_path), read_docs(backup_path)
    for elastic_key, elastic_value in elasticsearch_docs.items():
        for backup_key, backup_value in backup_docs.items():
            if backup_key.startswith(elastic_key) or elastic_key.startswith(backup_key):
                print("Found possible similar documents")
                print("Backup title: {}".format(backup_value[0]))
                print("Elastic title: {}".format(elastic_value[0]))
                if backup_value[1] != elastic_value[1]:
                    print("Possible similar documents do not have the same content")
                    print("Backup title: {}".format(backup_value[0]))
                    print("Elastic title: {}".format(elastic_value[0]))


def clear_elastic():
    from elasticsearch import Elasticsearch
    from elasticsearch_dsl import Search, Q
    client = Elasticsearch([{
        "host": "143.233.226.60",
        "port": 9200,
        "http_auth": ("debatelab", "SRr4TqV9rPjfzxUmYcjR4R92")
    }], timeout=60)
    s = Search(using=client, index="debatelab").query("match_all").delete()


if __name__ == '__main__':
    # main()
    clear_elastic()