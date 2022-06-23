import random
import itertools
import sys
from collections import defaultdict
import re
import os
from glob import glob
from ellogon import tokeniser
from dataclasses import dataclass
from src.utils import align_expanded_tokens, tokenize_with_spans

essays_folder = sys.argv[1] if len(sys.argv) > 1 else "essays_data"

adu_overlap_handling = "random_tie_breaks"
multiple_major_claim_handling = "random_tie_breaks"
keep_docs_with_atleast_one_adu_per_type = True

adu_types = "Claim MajorClaim Premise".split()
max_negative_rel_per_file = 10


@dataclass
class ADU:
    start: int
    end: int
    rng: range
    text: str
    id: str
    type: str


@dataclass
class Relation:
    type: str
    src: str
    trg: str


@dataclass
class Stance:
    id: str
    adu: str
    type: str


def is_adu(txt):
    return any(txt.startswith(x) for x in "Claim MajorClaim Premise".split())


def is_rel(txt):
    return any(txt.startswith(x) for x in "supports attacks".split())


def is_stance(txt):
    return any(txt.startswith(x) for x in "Stance".split())


def fix_annotation_offsets(text, adus, overlap_handling="random_tie_breaks", multimatch_handling="random_tie_breaks"):
    # raw_tokens, expanded_tokens = tokenize_with_spans(text)
    for adu_type, adus_list in adus.items():
        aligned_adus = []
        for adu in adus_list:
            match = list(re.finditer(re.escape(adu.text), text))
            if len(match) != 1:
                rmatch = []
                for ma in match:
                    if text[ma.regs[0][0]:ma.regs[0][1]] == adu.text:
                        rmatch.append(ma)
                match = rmatch
                # match = random.sample(match, 1)
                # raise ValueError("Adu text mismatch!")
            if not match:
                # nothing found, skip
                print("WARNING: skipping adu due to no match found for:", adu.id, adu.text)
                continue
            match = match[0]
            if len(match.regs) != 1:
                raise ValueError("Adu text mismatch!")
            start, end = match.regs[0]
            assert text[start:end] == adu.text, "Adu text mismatch!"
            aligned_adus.append(ADU(start, end, range(start, end), adu.text, adu.id, adu_type))
        adus[adu_type] = aligned_adus

    if overlap_handling == "random_tie_breaks":
        # either way, prefer MC adus
        mc = adus["MajorClaim"]
        assert len(mc) == 1, "Multiple Major Claims encountered in offset correction!"
        mc = mc[0]
        mc_idxs = mc.rng

        for other_adu_type in [k for k in adus if k != "MajorClaim"]:
            data = adus[other_adu_type]
            no_mc_overlaps = [x for x in data if not any(i in mc_idxs for i in x.rng)]
            adus[other_adu_type] = no_mc_overlaps
            if len(data) != len(no_mc_overlaps):
                print(f"Keeping {len(no_mc_overlaps)} / {len(data)} instances of {other_adu_type} due to MC overlap")

        # resolve ties between non-MC adus
        # fetch
        resolved = defaultdict(list)
        other_adus = [a for x in adus.values() for a in x if a.type != "MajorClaim"]
        # shuffle
        other_adus = random.sample(other_adus, len(other_adus))

        while other_adus:
            # for each adu
            current = other_adus.pop(0)
            # keep it
            resolved[current.type].append(current)

            non_overlapping = []
            for other_adu in other_adus:
                # for each other adu, if there's no conflict, keep it
                if not any(k in current.rng for k in other_adu.rng):
                    non_overlapping.append(other_adu)
                else:
                    # if there is, discard it
                    print("Overlapping adus -- keeping the first:")
                    print(1, current.id, current)
                    print(2, other_adu.id, other_adu)
            other_adus = non_overlapping

        resolved["MajorClaim"].append(mc)
        adus = resolved
    return adus


annot = defaultdict(list)
annot_files = glob(essays_folder + "*.ann")
annot_counts = {k: 0 for k in "adu rel stance".split()}
for p, path in enumerate(annot_files):
    doc_id = os.path.basename(path)
    with open(path) as f:
        annotations = [x.strip() for x in f.readlines()]
    print(f"Populating adus for file {p + 1} / {len(annot_files)}:", path)
    adus, rels, stances = defaultdict(list), [], []
    for ann in annotations:
        data = ann.split("\t")
        id_ = data.pop(0)
        type_ = data.pop(0)
        if is_adu(type_):
            # offsets in the dataset are wrong (pertaining to pre-translation offsets)
            # just save the text to align later
            adu_type, start, end = type_.split()
            start, end = int(start), int(end)
            if not data:
                # there are some errors in the annotations; skip them
                continue
            text = data.pop(0)
            adu = ADU(start, end, range(start, end), text, id_, adu_type)
            adus[adu_type].append(adu)
        elif is_rel(type_):
            rel_type, arg1, arg2 = type_.split()
            rels.append(Relation(rel_type, arg1[5:], arg2[5:]))
        elif is_stance(type_):
            _, target_adu, stance_type = type_.split()
            stances.append(Stance(id_, target_adu, stance_type))
        else:
            raise ValueError("Unresolved annotation")

    if keep_docs_with_atleast_one_adu_per_type:
        empty_adus = any(len(adus[k]) == 0 for k in "Claim Premise MajorClaim".split())
        if empty_adus:
            print("Omitting doc", path, "since they lack at least one type of ADUs")
            continue
    if multiple_major_claim_handling == "random_tie_breaks":
        print("Randomly breaking ties on multiple major claims.")
        adus["MajorClaim"] = random.sample(adus["MajorClaim"], 1)
    txt = os.path.splitext(path)[0] + ".txt"
    with open(txt) as f:
        txt = f.read()

    adus_dict = fix_annotation_offsets(txt, adus, overlap_handling=adu_overlap_handling)
    # adus = [v for (k, vals) in adus_dict.items() for v in ([k] + list(vals))]
    adus = [x for (k, v) in adus_dict.items() for x in v]
    adu_ids = [a.id for a in adus]
    adu_annotation_rngs = [x.rng for x in adus]

    # discard relations / stances on discarded adus
    rels = [r for r in rels if (r.src in adu_ids and r.trg in adu_ids)]
    stances = [s for s in stances if s.adu in adu_ids]

    # tokenize raw texts
    raw_sentences = tokeniser.tokenise_spans(txt)
    # flatten
    prev_label = None
    # ADUS
    for (sentence, sstart, send) in raw_sentences:
        for (token, tstart, tend) in sentence:
            # token is part of annotation
            adu_idx = [idx for idx, adu_rng in enumerate(adu_annotation_rngs) if
                       tstart in adu_rng and tend in adu_rng]
            if not adu_idx:
                prefix, label, mark = '', 'O', 'N'
                if label != prev_label:
                    print("\nO: ", end="")
            else:
                assert len(
                    adu_idx) == 1, f"Token {(token, tstart, tend)} found in many annotations: {[adus[x] for x in adu_idx]}!"
                label = adus[adu_idx[0]].type
                if label == prev_label:
                    prefix = "I-"
                else:
                    prefix = "B-"
                    print(label, ":", end="")
                prefix, label, mark = '', label, 'Y'
            if label not in adu_types + ["O"]:
                print()
            print(f"[{token}]", end="-")
            annot['adu'].append(f'{token}\t{prefix}{label}\t{mark}\tDoc: {doc_id}\n')
            prev_label = label
            print(end="")
        print()
        annot['adu'].append(f'\n')
        annot_counts['adu'] += 1
    # REL
    mapping = {r.src: (r.trg, r.type) for r in rels}
    # labels, sources, targets = list(zip(*rels))
    # mapping = dict(zip(sources, zip(targets, labels)))
    adu_pairs = list(itertools.product(adus, repeat=2))
    random.shuffle(adu_pairs)
    for c, (adu1, adu2) in enumerate(adu_pairs):
        # if the pair is not in the rel mapping, add it as a negative
        id1, id2 = adu1.id, adu2.id
        t1, t2 = adu1.text, adu2.text
        if id1 in mapping and mapping[id1][0] == id2:
            label = mapping[id1][1]
        else:
            label = "other"
        annot['rel'].append(f"{t1} [SEP] {t2}\t{label}\tPair: {len(annot['rel'])}\n")
        annot_counts['rel'] += 1
        if c > max_negative_rel_per_file:
            break

    # STANCE
    major_claim_text = adus_dict["MajorClaim"][0].text
    for sta in stances:
        try:
            matched_claim = [x for x in adus_dict["Claim"] if x.id == sta.adu][0]
        except IndexError:
            print(f"Could not find adu: {sta.adu} for stance {sta.id}")
        adu_text = matched_claim.text
        matched_adus = [x for x in adus_dict["Claim"] if x.id == sta.adu]
        assert len(matched_adus) == 1, f"Multiple adus match to stance {sta}!"
        adu_text = matched_adus[0].text
        annot['stance'].append(f"{adu_text} [SEP] {major_claim_text}\{sta.type}\tPair: {len(annot['stance'])}\n")
        annot_counts['stance'] += 1

print("Final annotation counts:")
for k, v in annot_counts.items():
    print(k, v, f"avg. ({round(v / len(annot_files), 2)} per file)")

output_folder = "essays_data"
os.makedirs(output_folder, exist_ok=True)
for ttype in annot:
    with open(f"{output_folder}/{ttype}.csv", "w") as f: \
            f.writelines(annot[ttype])
