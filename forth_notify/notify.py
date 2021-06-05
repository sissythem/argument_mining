import base64
import json

import requests

document_ids = ['baadf1a74546e9e1bc377b1c4304d67428003895', '614cfba661d2e321836e5e84f30bf345184820b6',
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
url = "https://isl.ics.forth.gr/debatelab_mq/api/exchanges/dlab/amq.default/publish"
username = "debateLab3ServerUser"
password = "debateLabPassw0rd"
data = {"properties": {"delivery_mode": 2}, "routing_key": "dlabqueue", "payload": json.dumps(document_ids),
        "payload_encoding": "string"}
creds = f"{username}:{password}"
creds_bytes = creds.encode("ascii")
base64_bytes = base64.b64encode(creds_bytes)
base64_msg = base64_bytes.decode("ascii")
headers = {"Content-Type": "application/json", "Authorization": f"Basic {base64_msg}"}
try:
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Request to ICS was successful!")
    else:
        print(f"Request to ICS failed with status code: {response.status_code} and message:{response.text}")
except(BaseException, Exception) as e:
    print(f"Request to ICS failed: {e}")
