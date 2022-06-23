import json
from collections import defaultdict
from os import makedirs
import shutil
from os.path import join
from ellogon import tokeniser
from tracking import AnnotationTracker
import itertools

from generation_utils import *

path = "/home/nik/work/debatelab/argument-mining/app/resources/21_docs.json"
collection_name = "kasteli-fixed"
kasteli_32_name = "DebateLab 1 Kasteli"
# collection_name = "Manual-22-docs-covid-crete-politics"
# collection_name = kasteli_32_name
collection_name = 'CollectedDocuments'

# read_source = "file"
read_source = "tool"

newline_norm = False
text_preproc = None
text_preproc = "newline_norm"

# min tracking token mismatch length
min_tracking_error = 50
# min num tokens for stance/ relations negatives
max_pairwise_negatives_ratio = 3
use_subsequence_negatives = False
min_num_tok_negatives = 10

omit = [
    "Ομιλία του Πρωθυπουργού Κυριάκου Μητσοτάκη στη Βουλή στη συζήτηση για τη διαχείριση των καταστροφικών πυρκαγιών και τα μέτρα αποκατάστασης"]

#########################
adu_results, rel_results, stance_results = [], [], []

if collection_name == 'kasteli-fixed':
    with open('/home/nik/work/debatelab/argument-mining/app/resources/annotations/kasteli-fixed.json') as f:
        jj = json.load(f)

sentence_id = 0
if read_source == "file":
    documents = read_from_file(path)
elif read_source == "tool":
    documents = read_from_annotation_tool(collection_name)

num_negatives_rel = []
num_negatives_stance = []
num_raw_annots = []
num_annots = []
num_total_duplicate_relations = 0
num_total_duplicate_stances = 0
add_negatives = True

# documents = list(sorted(documents, key=lambda x: x.name))

for doc_idx, doc in enumerate(documents):
    if omit:
        if doc.external_name in omit:
            print(f"OMITTING doc {doc_idx + 1} / {len(documents)}:", doc.external_name)
            continue
    # Simulate how codemirror handles lines...
    # text_cm = "\n".join(doc.text.splitlines())
    text_cm = doc.text
    if text_preproc == "newline_norm":
        text_cm = "\n".join(text_cm.splitlines())
    if text_preproc == "squash":
        text_cm = "".join(text_cm.splitlines())

    annotations, annotation_set = dict(), []
    rel_annotations, stance_annotations = [], []
    rel_mapping, stance_mapping, dangling_relations = {}, {}, defaultdict(list)
    sp = 0
    # Collect ADU annotations...
    argument_adus = list(doc.annotationsByType('argument'))
    argument_adus = [a for a in argument_adus if a.is_valid()]
    for ann in argument_adus:
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
            found = False
            off = 80
            for os in range(-off, off):
                for oe in range(-off, off):
                    ss = s + os
                    ee = e + oe
                    if segment_annot == text_cm[ss: ee]:
                        print("Fixed start:", ss)
                        print("Fixed end:", ee)

                        if collection_name == 'kasteli-fixed':
                            ann_json = [x for x in jj['data']['documents'][doc_idx]['annotations'] if
                                        x['_id'] == ann._id]
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
    arg_relations = [x for x in doc.annotationsByType('argument_relation') if x.is_valid()]
    num_annot = 0
    num_raw_annots.append(len(arg_relations))

    for ann in arg_relations:
        if collection_name == kasteli_32_name:
            # if ann.annotator_id != 'Button_Annotator_neutral_argument_type_Generic':
            if ann.annotator_id is None or ann.annotator_id.strip() == "":
                continue
        num_annot += 1
        if len(ann.spans) == 0:
            # Ensure that all relation annotations exist!
            arg1, arg2 = ann.attributeGet('arg1'), ann.attributeGet('arg2')
            try:
                assert (arg1.value in annotations) and (arg2.value in annotations)
            except AssertionError:
                dangling_relations[ann._id].append(ann)
                continue
        # get relation type
        reltype = ann.attributeGet("type").value
        relation, *stance = reltype.split("-")
        if collection_name == kasteli_32_name:
            if relation in ("for", "against"):
                # it's a stance
                relation, stance = None, relation

        arg1, arg2 = ann.attributeGet('arg1').value, ann.attributeGet('arg2').value
        # get corresponding segments
        s1, s2 = annotations[arg1].spans[0].segment, annotations[arg2].spans[0].segment
        i1 = range(int(annotations[arg1].spans[0].start), int(annotations[arg1].spans[0].end))
        i2 = range(int(annotations[arg2].spans[0].start), int(annotations[arg2].spans[0].end))

        s1 = s1.replace("\n", " ")
        s2 = s2.replace("\n", " ")
        if relation is not None:
            rel_results.append(f" {s1} [SEP] {s2}\t{relation}\tPair: {len(rel_results)}\n")
            rel_annotations.extend([i1, i2])
            try:
                assert not (s1 in rel_mapping and rel_mapping[s1] == s2), "Duplicate key arg in rel!"
            except AssertionError:
                print()
                num_total_duplicate_relations += 1
            rel_mapping[s1] = s2
        if stance:
            stance = stance[0] if isinstance(stance, list) else stance
            stance_results.append(f"{s1} [SEP] {s2}\t{stance}\tPair: {len(stance_results)}\n")
            stance_annotations.extend([i1, i2])
            try:
                assert not (s1 in stance_mapping and stance_mapping[s1] == s2), "Duplicate key arg in stance!"
            except AssertionError:
                print()
                num_total_duplicate_stances += 1

            stance_mapping[s1] = s2
        # mark offset ranges with relation annotations
    num_annots.append(num_annot)

    arg = [(x.attributes[0].value, [l[0] for l in w[0]]) for x in argument_adus for w in
           tokeniser.tokenise_spans(x.spans[0].segment)]
    # keep track of annotations
    annotation_tokens = [[w[0] for s in tokeniser.tokenise_spans(x.spans[0].segment) for w in s[0]] for x in
                         argument_adus]

    tracker = AnnotationTracker(annotation_tokens, min_tracking_error=min_tracking_error)

    if dangling_relations:
        print("Doc:", doc.external_name, len(dangling_relations), "dangling relations")
    relations_neg, stances_neg = [], []
    tokenization = tokeniser.tokenise_spans(text_cm)
    raw_tokenization = [[x[0] for x in y[0]] for y in tokenization]
    for sent_idx, (tokens, sent_start, sent_end) in enumerate(tokenization):
        raw_tokens = [x[0] for x in tokens]
        # print("Processing tokens:")
        # print(raw_tokens)
        prev_label = 'O'
        token_in_rel, token_in_stance = [], []
        token_group = []
        for token_idx, (token, start, end) in enumerate(tokens):
            # mark whether the token participates in stance / rel positives
            token_in_rel.append(any(start in r or end in r for r in rel_annotations))
            token_in_stance.append(any(start in r or end in r for r in stance_annotations))

            label = token_label(annotation_set, int(start), int(end))
            # print(f"Token [{token}] -> label: ", label)
            if label is not None:
                if label == 'O':
                    prefix = ''
                    mark = 'N'
                else:
                    try:
                        tracker.advance(token)
                    except ValueError:
                        # unable to track sequence, probably due to a tokenization error
                        continue

                    if label == prev_label:
                        # if the token is at the beginning of an annotation, break off the adu even if it's not at the
                        # beginning of the sentence. This handles cases of multiple adus within a sentence
                        if token_at_annotation_start(annotation_set, int(start)):
                            # B-annotation
                            prefix = 'B-'
                            mark = 'Y'
                        else:
                            # I-annotation
                            prefix = 'I-'
                            mark = 'Y'
                    else:
                        prefix = 'B-'
                        mark = 'Y'
                if token == '"':
                    token = '""""'
                # print(f'{token}\t{prefix}{label}\t{mark}\tSP: {sp}\tSentence: {sentence_id}\tDoc: {doc.id}')
                token_group.append(
                    f'{token}\t{prefix}{label}\t{mark}\tDoc: {doc.id}\n')
                sp += 1
                prev_label = label
            else:
                break
        # end of sentence, add blank line
        tracker.finish_tracking_sequence()
        # adu_results.append(f"\t\t\t Doc: {doc.id}\n")
        adu_results.extend(token_group)
        adu_results.append(f"\n")

        if add_negatives and use_subsequence_negatives:
            # eligible negative for rel / stance: no token has to participate
            rcont, idxs = split_to_contiguous(token_in_rel)
            for k, ix in zip(rcont, idxs):
                if len(k) < min_num_tok_negatives:
                    continue
                if not any(k):
                    s, e = ix[0], ix[-1]
                    relations_neg.append(" ".join(x[0] for x in tokens[s:e + 1]))
            scont, idxs = split_to_contiguous(token_in_stance)
            for k, ix in zip(scont, idxs):
                if len(k) < min_num_tok_negatives:
                    continue
                if not any(k):
                    s, e = ix[0], ix[-1]
                    stances_neg.append(" ".join(x[0] for x in tokens[s:e + 1]))
        sentence_id += 1
    # end of token sequence loop

    tracker.check_sanity()

    if add_negatives:
        adus = list(rel_mapping.items()) + list(stance_mapping.items())
        all_adus = set([x for t in adus for x in t] + stances_neg + relations_neg, )
        print("Adding REL negatives")
        prev = len(rel_results)
        rel_results = add_mapping_as_negatives(all_adus, rel_mapping, rel_results,
                                               max_num_append=int(max_pairwise_negatives_ratio * len(rel_mapping)))
        num_negatives_rel.append(len(rel_results) - prev)
        print("Will not add STANCE negatives, as they're not defined in our setting")
        # stance_results = add_mapping_as_negatives(all_adus, stance_mapping, stance_results, max_num_append=int(max_pairwise_negatives_ratio * len(stance_mapping)))

    if use_subsequence_negatives:
        # populate rel / stance negatives from non-relation tokens
        n = 0
        for s1, s2 in itertools.product(relations_neg, repeat=2):
            rel_results.append(f"{s1} [SEP] {s2}\tother\tPair: {len(rel_results)}\n")
            n += 1
        print("Added", n, "relation negatives")
        # n = 0
        # for s1, s2 in itertools.product(stances_neg, repeat=2):
        #     stance_results.append(f"{s1} [SEP] {s2}\tother\tPair: {len(stance_results)}\n")
        #     n += 1
        # print("Added", n, "stance negatives")
for coll in rel_results, stance_results:
    try:
        assert len(coll) == len(set(coll)), "Duplicates found!"
    except AssertionError:
        print()
print("Duplicate REL / STANCE:", num_total_duplicate_relations, num_total_duplicate_stances)
print("Num. raw annotations per doc:", num_raw_annots)
print("Num. annotations per doc:", num_annots)
print("Total num of REL negatives:", sum(num_negatives_rel))
makedirs(join(collection_name, "connl_data/adu"), exist_ok=True)
makedirs(join(collection_name, "connl_data/rel"), exist_ok=True)
makedirs(join(collection_name, "connl_data/stance"), exist_ok=True)
output_path = collection_name
with open(join(output_path, "connl_data", "adu", "train.csv"), "w") as f:
    print("Writing ADU dataset to ", f.name, len(adu_results), "data")
    f.writelines([x for x in adu_results])
for dd in "dev test".split():
    shutil.copy(join(output_path, "connl_data", "adu", "train.csv"),
                join(output_path, "connl_data", "adu", dd + ".csv"))
with open(join(output_path, "connl_data", "rel", "train.csv"), "w") as f:
    print("Writing REL dataset to ", f.name, len(rel_results), "data")
    f.writelines([x for x in rel_results])
for dd in "dev test".split():
    shutil.copy(join(output_path, "connl_data", "rel", "train.csv"),
                join(output_path, "connl_data", "rel", dd + ".csv"))
with open(join(output_path, "connl_data", "stance", "train.csv"), "w") as f:
    print("Writing STA dataset to ", f.name, len(stance_results), "data")
    f.writelines([x for x in stance_results])
for dd in "dev test".split():
    shutil.copy(join(output_path, "connl_data", "stance", "train.csv"),
                join(output_path, "connl_data", "stance", dd + ".csv"))
