import json
from collections import defaultdict
import os
import shutil
from os.path import join
from api import Session, Collection, Document, Span, Attribute
from ellogon import tokeniser
import itertools
from app.src.utils import utils

path = "/home/nik/work/debatelab/argument-mining/app/resources/21_docs.json"
kasteli_32_name = "DebateLab 1 Kasteli"
collection_name = kasteli_32_name
collection_name = "kasteli-fixed"
collection_name = "Manual-22-docs-covid-crete-politics"

read_source = "file"
read_source = "tool"
newline_norm = False
text_preproc = None
text_preproc = "newline_norm"

# min num tokens for stance/ relations negatives
min_num_tok_negatives = 10

omit = ["Ομιλία του Πρωθυπουργού Κυριάκου Μητσοτάκη στη Βουλή στη συζήτηση για τη διαχείριση των καταστροφικών πυρκαγιών και τα μέτρα αποκατάστασης"]

#########################
adu_results = []
rel_results = []
stance_results = []

def token_in_annotations(annotation_set, start, end):
    for ann in annotation_set:
        if ann.spansIncludeOffsets(start, end):
            return ann
    return None

def token_label(annotation_set, start, end):
    ann = token_in_annotations(annotation_set, start, end)
    if ann:
        return ann.attributeGet('type').value
    return 'O'

def read_from_file(path):
    with open(path) as f:
        return json.load(f)

def read_from_annotation_tool(collection_name):
    credentials_file = "credentials.json"
    with open(credentials_file) as f:
        creds = json.load(f)
    session = Session(**creds)

    ##
    # Get a Collection...
    ##
    col = session.collectionGetByName(collection_name)
    docs = list(col.documents())
    session.logout()
    return docs

if collection_name == 'kasteli-fixed':
    with open('/home/nik/work/debatelab/argument-mining/app/resources/annotations/kasteli-fixed.json') as f:
        jj = json.load(f)

sentence_id = 0
if read_source == "file":
    documents = read_from_file(path)
elif read_source == "tool":
    documents = read_from_annotation_tool(collection_name)


def split_to_contiguous(llist):
    curr_val = None
    ret = []
    idxs = []
    for i, x in enumerate(llist):
        if x != curr_val:
            ret.append([])
            idxs.append([])
            curr_val = x

        ret[-1].append(x)
        idxs[-1].append(i)
    return ret, idxs


for doc_idx, doc in enumerate(documents):
    if omit:
        if doc.external_name in omit:
            print(f"OMITTING doc {doc_idx+1} / {len(documents)}:", doc.external_name)
            continue
    # Simulate how codemirror handles lines...
    # text_cm = "\n".join(doc.text.splitlines())
    text_cm = doc.text
    if text_preproc == "newline_norm":
        text_cm = utils.normalize_newlines(text_cm)
    if text_preproc == "squash":
        text_cm = "".join(text_cm.splitlines())

    annotations = dict()

    rel_annotations = []
    stance_annotations = []

    rel_mapping = defaultdict(list)
    stance_mapping = defaultdict(list)

    dangling_relations = defaultdict(list)
    annotation_set = []
    sp = 0
    # Collect ADU annotations...
    adus = list(doc.annotationsByType('argument'))
    for ann in adus:
        first_span = ann.spans[0]
        s, e = int(first_span.start), int(first_span.end)
        segment_cm = text_cm[s: e]
        segment_annot = ann.spans[0].segment

        try:
            assert segment_cm == segment_annot
        except AssertionError as ae:
            print("Unequal segments!")
            print("Ann. id:", ann._id)
            print(segment_cm)
            print(segment_annot)
            found=False
            off = 80
            for os in range(-off, off):
                for oe in range(-off, off):
                    ss = s + os
                    ee = e + oe
                    if segment_annot == text_cm[ss: ee]:
                        print("Fixed start:", ss)
                        print("Fixed end:", ee)

                        if collection_name == 'kasteli-fixed':
                            ann_json = [x for x in jj['data']['documents'][doc_idx]['annotations'] if x['_id'] == ann._id]
                            assert len(ann_json) == 1
                            ann_json = ann_json[0]
                            ann_json["spans"][0]['start'] = ss
                            ann_json["spans"][0]['end'] = ee
                        found = True
                        break
                if found:
                    break
            if not found:
                print("Failed to resolve index!")

        annotations[ann._id] = ann
        annotation_set.append(ann)

    # Get all relations, and make sure we have all ADUs...
    for ann in doc.annotationsByType('argument_relation'):
        if collection_name == kasteli_32_name:
            if ann.annotator_id != 'Button_Annotator_neutral_argument_type_Generic':
                continue
        if len(ann.spans) == 0:
            # Relations...
            # Ensure that all annotations exist!
            arg1 = ann.attributeGet('arg1')
            arg2 = ann.attributeGet('arg2')
            try:
                assert arg2.value in annotations
                assert arg1.value in annotations
            except AssertionError:
                dangling_relations[ann._id].append(ann)
                continue
        # get relation type
        reltype = ann.attributeGet("type").value
        relation, *stance = reltype.split("-")
        arg1, arg2 = ann.attributeGet('arg1').value, ann.attributeGet('arg2').value
        # get corresponding segments
        s1, s2 = annotations[arg1].spans[0].segment, annotations[arg2].spans[0].segment
        rel_results.append(f"[CLS] {s1}yy [SEP] {s2}\t{relation}\tPair: {len(rel_results)}\n")
        i1 = range(int(annotations[arg1].spans[0].start), int(annotations[arg1].spans[0].end))
        i2 = range(int(annotations[arg2].spans[0].start), int(annotations[arg2].spans[0].end))
        rel_annotations.extend([i1, i2])
        rel_mapping[s1].append(s2)
        if stance:
            stance = stance[0]
            stance_results.append(f"[CLS]{s1} [SEP] {s2}\t{stance}\tPair: {len(stance_results)}\n")
            stance_annotations.extend([i1, i2])
            stance_mapping[s1].append(s2)
        # mark offset ranges with relation annotations


    if dangling_relations:
        print("Doc:", doc.external_name, len(dangling_relations), "dangling relations")
    relations_neg = []
    stances_neg = []
    for tokens, sent_start, sent_end in tokeniser.tokenise_spans(text_cm):
        prev_label = 'O'
        token_in_rel = []
        token_in_stance = []
        for token, start, end in tokens:

            token_in_rel.append(any(start in r or end in r for r in rel_annotations))
            token_in_stance.append(any(start in r or end in r for r in stance_annotations))

            label = token_label(annotation_set, int(start), int(end))
            if label == 'O':
                prefix = ''
                mark = 'N'
            elif label == prev_label:
                prefix = 'I-'
                mark = 'Y'
            else:
                prefix = 'B-'
                mark = 'Y'
            if token == '"':
                token = '""""'
            # print(f'{token}\t{prefix}{label}\t{mark}\tSP: {sp}\tSentence: {sentence_id}\tDoc: {doc.id}')
            adu_results.append(
                f'{token}\t{prefix}{label}\t{mark}\tDoc: {doc.id}\n')
            sp += 1
            prev_label = label
        # end of sentence, add blank line
        adu_results.append(f"\t\t\t Doc: {doc.id}\n")

        # eligible negative for rel / stance: no token has to participate
        rcont, idxs = split_to_contiguous(token_in_rel)
        for k, ix in zip(rcont, idxs):
            if len(k) < min_num_tok_negatives:
                continue
            if not any(k):
                s, e = ix[0], ix[-1]
                relations_neg.append(" ".join(x[0] for x in tokens[s:e+1]))
        scont, idxs = split_to_contiguous(token_in_stance)
        for k, ix in zip(scont, idxs):
            if len(k) < min_num_tok_negatives:
                continue
            if not any(k):
                s, e = ix[0], ix[-1]
                stances_neg.append(" ".join(x[0] for x in tokens[s:e+1]))


        sentence_id += 1

    # populate negatives from positives
    for k, vlist in rel_mapping.items():
        # map each value to the key, if not a v->k relation already exists
        for v in vlist:
            if not (v in rel_mapping and k in rel_mapping[v]):
                s1, s2 = v, k
                rel_results.append(f"[CLS]{s1} [SEP] {s2}\tother\tPair: {len(rel_results)}\n")
    for k, vlist in stance_mapping.items():
        # map each value to the key, if not a v->k relation already exists
        for v in vlist:
            if not (v in stance_mapping and k in stance_mapping[v]):
                s1, s2 = v, k
                stance_results.append(f"[CLS]{s1} [SEP] {s2}\tother\tPair: {len(stance_results)}\n")

    # populate rel / stance negatives from non-relation tokens
    n = 0
    for s1, s2 in itertools.product(relations_neg, repeat=2):
        rel_results.append(f"[CLS]{s1} [SEP] {s2}\tother\tPair: {len(rel_results)}\n")
        n += 1
    n = 0
    print("Added", n, "relation negatives")
    for s1, s2 in itertools.product(stances_neg, repeat=2):
        stance_results.append(f"[CLS]{s1} [SEP] {s2}\tother\tPair: {len(stance_results)}\n")
        n += 1
    print("Added", n, "stance negatives")

# print(tokeniser.tokenise_spans(text_cm))
os.makedirs(join(collection_name, "connl_data/adu"))
os.makedirs(join(collection_name, "connl_data/rel"))
os.makedirs(join(collection_name, "connl_data/stance"))
output_path = collection_name
with open(join(output_path, "connl_data", "adu", "train.csv"), "w") as f:
    print("Writing ADU dataset to ", f.name, len(adu_results), "data")
    f.writelines(adu_results)
    for dd in "dev test".split():
        shutil.copy(join(output_path, "connl_data", "adu", "train.csv"),join(output_path, "connl_data", "adu", dd + ".csv") )
with open(join(output_path, "connl_data", "rel", "train.csv"), "w") as f:
    print("Writing REL dataset to ", f.name, len(rel_results), "data")
    f.writelines(rel_results)
    for dd in "dev test".split():
        shutil.copy(join(output_path, "connl_data", "rel", "train.csv"),
                    join(output_path, "connl_data", "rel", dd + ".csv"))
with open(join(output_path, "connl_data", "stance", "train.csv"), "w") as f:
    print("Writing STA dataset to ", f.name, len(stance_results), "data")
    f.writelines(stance_results)
    for dd in "dev test".split():
        shutil.copy(join(output_path, "connl_data", "stance", "train.csv"),
                    join(output_path, "connl_data", "stance", dd + ".csv"))
