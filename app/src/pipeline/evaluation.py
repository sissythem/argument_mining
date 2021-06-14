import json
from os import getcwd, listdir
from os.path import join
from pathlib import Path

from elasticsearch import Elasticsearch


def convert_annotations(document):
    annotations = {"ADUs": [], "Relations": []}
    adu_counter, rel_counter = 0, 0
    document_annotations = document["annotations"]
    for annotation in document_annotations:
        if annotation["type"] == "argument":
            adu_counter += 1
            annotation_type = ""
            for attr in annotation["attributes"]:
                if "name" in attr.keys():
                    type_attr = attr["name"]
                    if type_attr == "type":
                        annotation_type = attr["value"]
            adu = {
                "id": f"T{adu_counter}",
                "old_id": annotation["_id"],
                "type": annotation_type,
                "starts": annotation["spans"][0]["start"],
                "ends": annotation["spans"][0]["end"],
                "segment": annotation["spans"][0]["segment"]
            }
            annotations["ADUs"].append(adu)
    for annotation in document_annotations:
        if annotation["type"] == "argument_relation":
            rel_counter += 1
            annotation_attrs = annotation["attributes"]
            rel_type, arg1_id, arg2_id = "", "", ""
            for attr in annotation_attrs:
                attr_name = attr["name"]
                attr_value = attr["value"]
                if attr_name == "type":
                    rel_type = attr_value
                elif attr_name == "arg1":
                    arg1_id = attr_value
                elif attr_name == "arg2":
                    arg2_id = attr_value
            adu1, adu2 = get_adu_initial(adus=annotations["ADUs"], id1=arg1_id, id2=arg2_id)
            rel = {
                "id": f"R{rel_counter}",
                "type": rel_type,
                "arg1": adu1["id"],
                "arg2": adu2["id"],
            }
            annotations["Relations"].append(rel)
    document["annotations"] = annotations
    get_evidence_by_claim(relations=document["annotations"]["Relations"], sort=False)
    with open(join(path, "converted.json"), "w") as f:
        f.write(json.dumps(document, indent=4, sort_keys=False, ensure_ascii=False))


def get_adu_initial(adus, id1, id2):
    adu1, adu2 = {}, {}
    for adu in adus:
        if adu["old_id"] == id1:
            adu1 = adu
        elif adu["old_id"] == id2:
            adu2 = adu
    return adu1, adu2


def get_json_documents():
    # TODO change path
    documents = []
    path_to_folder = join(getcwd(), "json")
    files = listdir(path_to_folder)
    for json_file in files:
        file_path = join(path_to_folder, json_file)
        with open(file_path, "r") as f:
            content = json.loads(f.read())
        document = content["_source"]
        documents.append(document)
    return documents


def get_json_from_elasticsearch():
    documents = []
    elasticsearch_client = Elasticsearch([{
        'host': "143.233.226.60",
        'port': 9200,
        'http_auth': ("debatelab", "SRr4TqV9rPjfzxUmYcjR4R92")
    }], timeout=60)

    swo_ids = ['baadf1a74546e9e1bc377b1c4304d67428003895', '614cfba661d2e321836e5e84f30bf345184820b6',
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
    swo_ids = ['0a7e4aa11918ed0e3d294f4939c804244571863b']
    res = elasticsearch_client.mget(index="debatelab", body={"ids": swo_ids})
    res = res["docs"]
    for json_content in res:
        document = json_content["_source"]
        documents.append(document)
    return documents


def get_documents_from_folder(folder):
    filenames = listdir(folder)
    documents = []
    for filename in filenames:
        file_path = join(folder, filename)
        with open(file_path, "r") as f:
            data = json.loads(f.read())
            documents.append(data)
    return documents


def get_evidence_by_claim(relations, sort=True):
    rel_dict = {}
    for relation in relations:
        if relation["arg2"] not in rel_dict.keys():
            rel_dict[relation["arg2"]] = []
        rel_dict[relation["arg2"]].append(relation)
    if sort:
        for claim, relations in rel_dict.items():
            rel_dict[claim] = sorted(relations, key=lambda k: k['confidence'], reverse=True)
    return rel_dict


def get_adu(adus, adu_id, logs):
    counter = 0
    found_adu = None
    while counter < len(adus):
        adu = adus[counter]
        if adu["id"] == adu_id:
            found_adu = adu
            segment = adu["segment"]
            print(f"{adu['type']}: {segment}")
            logs.append(f"{adu['type']}: {segment}")
            break
        counter += 1
    return found_adu


def main():
    logs = []
    documents = get_json_from_elasticsearch()
    documents = get_documents_from_folder(folder=join(getcwd(), "json"))
    for document in documents:
        annotations = document["annotations"]
        adus = annotations["ADUs"]
        relations = annotations["Relations"]
        rel_dict = get_evidence_by_claim(relations=relations)
        for parent_arg, arg_rels in rel_dict.items():
            type_parent, type_children = "", ""
            adu = get_adu(adus=adus, adu_id=parent_arg, logs=logs)
            if adu:
                if adu["type"] == "claim":
                    type_parent = "claim"
                    type_children = "premises"
                else:
                    type_parent = "major claim"
                    type_children = "claims"
                    print(f"{type_parent} has {len(arg_rels)} {type_children}")
            logs.append(f"{type_parent} has {len(arg_rels)} {type_children}")
            print(f"Printing {type_children} for {type_parent}")
            logs.append(f"Printing {type_children} for {type_parent}")
            for rel in arg_rels:
                child = rel["arg1"]
                print(f"Relation type is {rel['type']}")
                logs.append(f"Relation type is {rel['type']}")
                get_adu(adus=adus, adu_id=child, logs=logs)


def main2():
    with open(join(path, "kasteli.json"), "r") as f:
        documents = json.load(f)
        documents = documents["data"]["documents"]
        for document in documents:
            if document["id"] == 202:
                convert_annotations(document=document)


if __name__ == '__main__':
    path = Path(__file__).parent.parent
    path = join(str(path), "resources", "data", "initial")
    main()
