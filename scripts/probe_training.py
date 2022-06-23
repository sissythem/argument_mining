import requests
import json
import sys
import pandas as pd

try:
    config = json.loads(sys.argv[1])
except IndexError:
    config = {}
try:
    access_token = sys.argv[2]
except IndexError:
    access_token = "123"
try:
    host_url = sys.argv[3]
    if host_url == "ncsr":
        host_url = "143.233.226.60"
except IndexError:
    host_url = 'localhost'

try:
    port = sys.argv[4]
except IndexError:
    port = '8000'

try:
    modelid = sys.argv[5]
except IndexError:
    modelid = "modelid"
print("Using modelid", modelid)

path = sys.argv[6]
with open(path) as f:
    data = f.read()

payload = {
    "data": data,
    "id": modelid,
    "config": json.dumps(config)
}
response = requests.post(
    f"http://{host_url}:{port}/train",
    json=payload,
    params={"access_token": access_token}
)
res = json.loads(response.content)
print(res)
