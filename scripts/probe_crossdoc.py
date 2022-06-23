import requests
import json
import sys


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

with open("pipeline_outputs/run_03042022_215856.json") as f:
    docs = json.load(f)['documents']

payload = {"docs": docs}
response = requests.post(
    f"http://{host_url}:8000/crossdocs",
    json=payload,
    params={"access_token": access_token}
)
res = json.loads(response.content)
print(res)
