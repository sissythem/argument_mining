import requests
from pprint import pprint
import json
import sys
import pandas as pd

try:
    access_token = sys.argv[1]
except IndexError:
    access_token = "123"

try:
    host_url = sys.argv[2]
    if host_url == "ncsr":
        host_url = "143.233.226.60"
except IndexError:
    host_url = 'localhost'

try:
    port = sys.argv[3]
except IndexError:
    port = '8080'

response = requests.get(
    f"http://{host_url}:{port}/models",
    params={"access_token": access_token}
)
res = json.loads(response.content)
pprint(res)
