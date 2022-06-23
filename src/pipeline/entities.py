import json
import logging
import requests
def get_named_entities(doc_id, content, ner_config):
    """
    Retrieves Named Entities using a RESTful API service

    Args
        doc_id (str): the id of the document
        content (str): the content of the document

    Returns
        list: the list of the entities found in the document
    """
    entities = []
    data = {"text": content, "doc_id": doc_id}
    url = ner_config["endpoint"]
    try:
        response = requests.post(url, data=json.dumps(data))
    except requests.exceptions.ConnectionError as ce:
        logging.error(f"Failed to connect to the entity provider: {str(ce)}")
        return []
    if response.status_code == 200:
        entities = json.loads(response.text)
    else:
        logging.error("Could not retrieve named entities")
        logging.error(
            f"Status code: {response.status_code}, reason: {response.text}")
    return entities