"""
Script to reset debatelab storage endpoints
"""
from src.io.elastic_io import ElasticSearchConfig
from src.utils import read_json_str_or_file
import sys
import json
import os
import logging
from src.utils import timestamp

logging.basicConfig(level=logging.INFO)
deletion_backup_folder = "deletion_backup"
os.makedirs(deletion_backup_folder, exist_ok=True)

config = read_json_str_or_file(sys.argv[1])['save']
es = ElasticSearchConfig(config)

# documents
docs = es.get_all_documents()
logging.info(f"Retrieved a total of {len(docs)} articles.")
with open(os.path.join(deletion_backup_folder, f"documents_{timestamp()}.json"), "w") as f:
    json.dump(docs, f)
es.delete_all_documents()
# for doc in docs:
#     es.delete_document(doc)
logging.info("Completed document deletion.")

rels = es.get_all_relations()
logging.info(f"Retrieved a total of {len(rels)} crossdoc relations...")

with open(os.path.join(deletion_backup_folder, f"relations_{timestamp()}.json"), "w") as f:
    json.dump(rels, f)

es.delete_all_relations()
logging.info("Completed crossdoc relation deletion.")
es.stop()
