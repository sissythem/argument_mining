import requests
from pprint import pprint
import json

# link = "https://www.efsyn.gr/politiki/exoteriki-politiki/332706_dendias-i-symfonia-gia-apostratiotikopoiisi-den-aforoyse-tin"
# link = "https://www.ekriti.gr/kriti-irakleio/arsenis-mnimeio-paralogismoy-neo-aerodromio-sto-kasteli-vinteo"
link = "https://www.metaforespress.gr/aeroporika/2025-%CF%83%CE%B5-%CE%BB%CE%B5%CE%B9%CF%84%CE%BF%CF%85%CF%81%CE%B3%CE%AF%CE%B1-%CF%84%CE%BF-%CE%BD%CE%AD%CE%BF-%CE%B1%CE%B5%CF%81%CE%BF%CE%B4%CF%81%CF%8C%CE%BC%CE%B9%CE%BF-%CF%83%CF%84%CE%BF-%CE%BA/"
payload = {
    "request": {
        "url": link
    },
    "spider_name": "genericurl",
    "crawl_args": {}
}
auth = tuple("debatelab:agyaijeynFev3ov5".split(":"))
args = {"url": "https://socialwebobservatory.iit.demokritos.gr/crawl.json", "json": payload}
if auth:
    args["auth"] = auth
response = requests.post(**args)
print(response)
pprint(response.status_code)
import pdb; pdb.set_trace()
