"""Script to process a list of retrieved documents
"""
import json
import os

#
# retrieved documents file to read
docs_path = "/home/nik/work/debatelab/argument-mining/app/output/output_files/retrieved_documents_2021-11-11_range_gt_2021-10-28,lte_2021-11-11_terms_Κρήτη,Σεισμός.json"
# list of document urls to exclude
exclude_links_path = "/home/nik/work/debatelab/argument-mining/app/resources/kasteli_34_urls.txt"
# where to write retained documents
retained_path = os.path.splitext(docs_path)[0] + "_retained.json"
#####################

links_to_exclude = []
if exclude_links_path is not None:
    with open(exclude_links_path) as f:
        links_to_exclude = []


with open(docs_path) as f:
    data = json.load(f)
retained = []
for i, dat in enumerate(data):
    print(i+1, "/", len(data), dat['title'], '\t', dat['link'])

    # keep document under these conditions
    if len(dat['content']) > 500 and \
            dat['link'] not in links_to_exclude and \
            dat['title'] not in [ret['title'] for ret in retained]:
        retained.append(dat)
print(f"Retained {len(retained)} docs.")
with open(retained_path, "w") as f:
    json.dump(retained, f)
