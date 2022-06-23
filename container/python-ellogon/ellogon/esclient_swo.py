# -*- coding: utf-8 -*-
from sshtunnel import SSHTunnelForwarder
from elasticsearch     import Elasticsearch
import threading

tunnel = SSHTunnelForwarder(
    ssh_address="socialwebobservatory.iit.demokritos.gr",
    ssh_port=222,
    ssh_username="petasis",
    ssh_private_key="/home/petasis/.ssh/id_rsa",
    remote_bind_address=('127.0.0.1', 9200),
    compression=True
)

tunnel.start()

elastic_server_client = Elasticsearch([{
        'host': 'localhost',
        'port': tunnel.local_bind_port,
        'http_auth':('elastic', 'fafOwgabdow.')
    }], timeout=60)

def stop():
    global elastic_server_client
    del elastic_server_client
    [t.close() for t in threading.enumerate() if t.__class__.__name__ == "Transport"]
    tunnel.stop()
