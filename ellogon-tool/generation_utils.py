import json
from api import Session, Collection, Document, Span, Attribute
import itertools
import random


def token_in_annotations(annotation_set, start, end):
    for ann in annotation_set:
        # this fails because agreement between tokenization and annotation is not guaranteed
        # e.g. an annotation might be: 'Τη θέση των δένδρων θα καταλάβουν τα κτήρια, οι χώροι προσγείωσης απογείωσης, περιβάλοντες δρόμοι, κτλ'
        # but tokenization yields: ['Τη', 'θέση', 'των',  ....  'δρόμοι', ',', 'κτλ.' ]
        # i.e. the last token's end index is always outside the annotation and always fails
        if ann.spansIncludeOffsets(start, end):
            return ann
    return None


def token_label(annotation_set, start, end):
    ann = token_in_annotations(annotation_set, start, end)
    if ann:
        ann_type = ann.attributeGet('type')
        # print("Matched token to annotation ", ann_type, ann.spans[0].segment)
        if ann_type is None:
            return None
        return ann_type.value
    return 'O'


def token_at_annotation_start(annotation_set, start):
    for ann in annotation_set:
        for span in ann.spans:
            if start == span.start:
                return True
    return False


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


def add_mapping_as_negatives(all_adus, existing_mappings, samples_container, max_num_append=None):
    # copy
    existing_mappings = list(existing_mappings.items())
    random.shuffle(existing_mappings)
    num_added = 0
    for s1, s2 in itertools.product(all_adus, repeat=2):
        # if the mapping exist (e.g. it's a positive or it's already added as a negative
        if (s1, s2) in existing_mappings:
            continue
        samples_container.append(f"{s1} [SEP] {s2}\tother\tPair: {len(samples_container)}\n")
        num_added += 1
        existing_mappings.append((s1, s2))
        if max_num_append is not None and num_added >= max_num_append:
            break
    print("Added", num_added, "negatives samples.")
    return samples_container
