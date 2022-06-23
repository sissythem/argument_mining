# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch

elastic_server_client = Elasticsearch([{
    'host': 'socialwebobservatory.iit.demokritos.gr',
    'host': 'localhost',
    'port': 9200,
    'http_auth':('elastic', 'fafOwgabdow.')
}])
