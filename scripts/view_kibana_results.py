"""
Process results from Kibana in an easy-to-inspect format
"""

import sys

import pandas
import json

pandas.set_option("max_columns", None)
pandas.set_option('max_colwidth', None)
pandas.set_option("expand_frame_repr", False)

csv_path = sys.argv[1]
data = pandas.read_csv(csv_path)

# get MCs
mcs = []
for adus in data['annotations.ADUs']:
    jd = json.loads(adus)
    m = [x for x in jd if x['type'] == "major_claim"]
    if len(m) != 1:
        for cc in m:
            print(cc['segment'])
    m = m[0]
    mcs.append(m['segment'])
data['major_claim'] = mcs

problematic_mcs = [
        "Η πρώτη εικόνα για την παραλλαγή Όμικρον:",
        ", διέθεσε 1",
        "Κάτοικοι Ευαγγελισμού για νέο αεροδρόμιο:"
        ]

mcdata = data["crawledAt title link major_claim".split()].sort_values("crawledAt link".split())
mcdata2 = data["title major_claim".split()]

mcprob = mcdata[mcdata["major_claim"].isin(problematic_mcs)]
print(len(mcdata), "articles")
import ipdb; ipdb.set_trace()
