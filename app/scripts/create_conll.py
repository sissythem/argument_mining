import json
from collections import defaultdict
from api import Session, Collection, Document, Span, Attribute
from ellogon import tokeniser
from app.src.utils import utils

path = "/home/nik/work/debatelab/argument-mining/app/resources/21_docs.json"
kasteli_32_name = "DebateLab 1 Kasteli"
collection_name = kasteli_32_name
collection_name = "Manual-22-docs-covid-crete-politics"

read_source = "file"
read_source = "tool"
do_newline_norm = False
omit = ["Ομιλία του Πρωθυπουργού Κυριάκου Μητσοτάκη στη Βουλή στη συζήτηση για τη διαχείριση των καταστροφικών πυρκαγιών και τα μέτρα αποκατάστασης"]

#########################
adu_results = []
rel_results = []
stance_results = []
output_path = collection_name

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

sentence_id = 0
if read_source == "file":
    documents = read_from_file(path)
elif read_source == "tool":
    documents = read_from_annotation_tool(collection_name)

for i, doc in enumerate(documents):
    if omit:
        if doc.external_name in omit:
            print(f"OMITTING doc {i+1} / {len(documents)}:", doc.external_name)
            continue
    # Simulate how codemirror handles lines...
    # text_cm = "\n".join(doc.text.splitlines())
    text_cm = doc.text
    if do_newline_norm:
        text_cm = utils.normalize_newlines(text_cm)

    annotations = dict()
    dangling_relations = defaultdict(list)
    annotation_set = []
    sp = 0
    # Collect ADU annotations...
    for ann in doc.annotationsByType('argument'):
        if collection_name == kasteli_32_name:
            if ann.annotator_id != 'Button_Annotator_neutral_argument_type_Generic':
                continue
        first_span = ann.spans[0]
        s, e = int(first_span.start), int(first_span.end)
        segment_cm = text_cm[s: e]
        assert segment_cm == ann.spans[0].segment
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
        rel_results.append(f"[CLS] {s1} [SEP] {s2}\t{relation}\tPair: {len(rel_results)}\n")
        if stance:
            stance = stance[0]
            stance_results.append(f"[CLS]{s1} [SEP] {s2}\t{stance}\tPair: {len(stance_results)}\n")


    if dangling_relations:
        print("Doc:", doc.external_name, len(dangling_relations), "dangling relations")
    for tokens, sent_start, sent_end in tokeniser.tokenise_spans(text_cm):
        prev_label = 'O'
        for token, start, end in tokens:
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
        sentence_id += 1
        # end of sentence, add blank line
        adu_results.append("")

# print(tokeniser.tokenise_spans(text_cm))
with open(output_path + "_adu.csv", "w") as f:
    print("Writing ADU dataset to ", f.name, len(adu_results), "data")
    f.writelines(adu_results)
with open(output_path + "_rel.csv", "w") as f:
    print("Writing REL dataset to ", f.name, len(rel_results), "data")
    f.writelines(rel_results)
with open(output_path + "_stance.csv", "w") as f:
    print("Writing REL dataset to ", f.name, len(stance_results), "data")
    f.writelines(stance_results)
