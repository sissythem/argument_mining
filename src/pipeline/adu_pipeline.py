import logging
import numpy as np
import random
from collections import defaultdict

from src.utils import tokenize_with_spans, join_sentences, get_sentence_raw_tokens, align_expanded_tokens, \
    tokens_to_text, join_sentence_tokens, make_segment_id, collect_adu_types


def run(documents, model):
    """
    Argument Mining pipeline:
    | 1. Predict ADUs for each document
    | 2. Predict relations & stance on the extracted ADUs
    | 3. Validate output json objects
    | 4. Perform corrections wherever necessary
    | 5. Save outputs in the DebateLab Elasticsearch

    Args
        | documents (list): list of documents extracted from the SocialObservatory Elasticsearch
        | export_schema (bool): True/False to extract json schema from the output json objects

    Returns
        tuple: the list of documents and the list of document ids of the **valid** json objects
    """
    # log confidence statistics
    results = []
    # extract ADUs
    for idx, document in enumerate(documents):
        doc_id = document['id']
        logging.info(
            f"Extracting ADUs for document {idx + 1}/{len(documents)}: {document['title']} -- {document['link']}")
        raw_segments = []
        tokenized, tokenized_raw = tokenize_with_spans(document['content'])
        for sent_idx, (expanded_tokens, tokens) in enumerate(zip(tokenized, tokenized_raw)):
            text_tokens = get_sentence_raw_tokens(tokens)
            original_text = tokens_to_text(expanded_tokens)
            labels, confidences = model.predict(text_tokens, original_text)
            created_segs = make_segments(text_tokens, expanded_tokens, labels, confidences)
            if created_segs:
                segs = process_segments(created_segs, tokens, expanded_tokens)
                raw_segments.extend(segs)
                for seg in segs:
                    if document['content'][seg["starts"]: seg["ends"]] != seg['segment']:
                        raise ValueError("Misaligned segment!")
        # all document sentences finished processing
        if not raw_segments:
            logging.error(f"No ADUs resolved at all for document {doc_id}")
            major_claims, claims, premises = [], [], []
        else:
            # add an id
            for s in raw_segments:
                s['id'] = make_segment_id()
            segments = check_major_claim(adus=raw_segments, first_sentence=(tokenized[0], tokenized_raw[0]))
            id_list = [x['id'] for x in segments]
            assert len(id_list) == len(set(id_list)), f"Duplicate segment id(s) after MC checks: {id_list}"
            major_claims, claims, premises = collect_adu_types(segments)

        # populate results
        logging.debug(f"Found {len(major_claims)} major claims, {len(claims)} claims and {len(premises)} premises")
        results.append({"major_claims": major_claims, "claims": claims, "premises": premises})
    flat_confidences_per_doc = [[float(v['confidence']) for vals in res.values() for v in vals] for res in results]
    flat_confidences = [v for vals in flat_confidences_per_doc for v in vals]
    logging.info(
        f"ADU pipeline confidence mean / median / std: {np.mean(flat_confidences):.2f} / {np.median(flat_confidences):.2f} / {np.std(flat_confidences):.2f}")
    return results


def process_segments(segments, sentence_tokens, sentence_expanded_tokens, sentence_offset=0):
    adus = []
    for s_idx, segment in enumerate(segments):
        logging.debug(f"Processing ADU detector output segment: {s_idx + 1}/{len(segments)}")
        # discard degenerate cases
        if len(segment['tokens']) <= 2:
            continue
        if not "".join(segment['tokens']) or not segment['label']:
            continue
        # if len(segment["text"]) == 1:
        #     continue

        logging.debug(f"Segment type: {segment['label']}")
        segment["source_sentences"] = [sentence_tokens]
        segment["source_sentences_expanded"] = [sentence_expanded_tokens]

        res = derive_segment_text(segment)
        if res is None:
            continue
        segment_text, start_idx, end_idx = res
        upd = {
            "type": segment["label"],
            "starts": start_idx,
            "ends": end_idx,
            "segment": segment_text,
            "tokens": segment['tokens'],
            "confidence": str(segment["confidence"])
        }
        segment.update(upd)
        adus.append(segment)
    return adus


def derive_segment_text(segment: dict):
    """
    Generates text for a segment, given its source expanded tokens & sentences

    Args:
        segment: A segment dict
    Returns:
        Generated text and its start / end index in the original content
    """
    # align segment tokens to expanded tokens
    # get flat list of expanded tokens
    exp_tokens = [tok for sent in segment["source_sentences_expanded"] for tok in sent[0]]
    exp_raw_tokens = [tok for sent in segment["source_sentences_expanded"] for tok in get_sentence_raw_tokens(sent)]
    s, e = align_expanded_tokens(segment["tokens"], exp_raw_tokens)
    start_idx = exp_tokens[s][-2]
    end_idx = exp_tokens[e][-1]
    if start_idx == -1 and end_idx == -1:
        return None

    matched_expanded_tokens = exp_raw_tokens[s: e + 1]
    segment_text = join_sentence_tokens(matched_expanded_tokens)
    return segment_text, start_idx, end_idx


def get_segment_offsets(segment, sentence_tokens):
    """Find segment start / end offsets

    :param segment: [description]
    :type segment: [type]
    :param sentence_tokens: [description]
    :type sentence_tokens: [type]
    :return: [description]
    :rtype: [type]
    """
    s, e = segment['token_indexes'][0], segment['token_indexes'][1]

    leading_tokens = sentence_tokens[:s]
    # plus on for spaces between
    start_offset = sum(len(x) + 1 for x in leading_tokens)
    end_offset = start_offset + sum(len(x) + 1 for x in segment['tokens']) - 2
    return start_offset, end_offset


def make_segments(tokens, exp_tokens, labels, confidences, null_label="O"):
    segments = []
    if tokens:
        idx = None
        while True:
            segment, idx = get_next_segment(tokens, labels, confidences, current_idx=idx)
            if segment:
                # consolidate tokens to text
                # locate edges to expanded seq
                # s, e = align_expanded_tokens(segment['text'], orig_tokenized)
                # reconstruct
                # text = "".join(orig_tokenized[s:e+1])
                # segment['text'] = join_sentences(segment['tokens'])
                segment["confidence"] = str(np.mean(segment["confidences"]))
                segments.append(segment)
            if idx is None:
                break
    return segments


def get_next_segment(tokens, labels, confidences, current_idx=None, current_label=None, segment=None):
    if current_idx is None:
        current_idx = 0
    if current_idx >= len(labels):
        return segment, None
    token, label, confidence = tokens[current_idx], labels[current_idx], confidences[current_idx]
    label_parts = label.split("-")
    if len(label_parts) > 1:
        label_type, label = label_parts[0], label_parts[1]
    else:
        label_type, label = None, label

    # if we're already tracking a contiguous segment:
    if current_label is not None:
        if label_type == "I" and current_label == label:
            # append to the running collections
            segment["confidences"].append(confidence)
            segment["tokens"].append(token)
            segment["token_indexes"].append(current_idx)
            return get_next_segment(tokens, labels, confidences, current_idx + 1, current_label, segment)
        else:
            # new segment, different than the current one
            # next function call should start at the current_idx
            return segment, current_idx
    else:
        # only care about B-tags to start a segment
        if label_type == "B":
            segment = {"label": label, "tokens": [token], "confidences": [confidence], "token_indexes": [current_idx]}
            return get_next_segment(tokens, labels, confidences, current_idx + 1, label, segment)
        else:
            return get_next_segment(tokens, labels, confidences, current_idx + 1, None, segment)


def check_major_claim(adus, first_sentence):
    if not any(s['type'] == "major_claim" for s in adus):
        # no mc found 
        adus, mc = handle_missing_major_claim(adus, first_sentence)
    else:
        major_claims, claims, premises = collect_adu_types(adus)
        other_adus = premises + claims
        if len(major_claims) == 1:
            # all good -- return originals
            adus, mc = other_adus, major_claims[0]
        else:
            adus, mc = handle_multiple_major_claims(major_claims=major_claims, adus=other_adus)
    adus.append(mc)
    return adus


def merge_adu_segments(a: dict, b: dict) -> dict:
    assert a['type'] == b['type'] and a["label"] == b["label"], "Attempted to merge two different-type ADUs!"
    joined_tokens = a['tokens'] + b['tokens']
    merged = {"id": make_segment_id(),
              "type": a["type"],
              "label": a["label"],
              "tokens": joined_tokens,
              "token_indexes": a["token_indexes"] + b["token_indexes"],
              "confidences": a['confidences'] + b['confidences'],
              "source_sentences": a['source_sentences'] + b['source_sentences'],
              "source_sentences_expanded": a['source_sentences_expanded'] + b['source_sentences_expanded'],
              "segment": a['segment'] + b['segment'],
              "starts": a['starts'], "ends": b["ends"],
              "confidence": str(np.mean([float(a['confidence']), float(b['confidence'])]))
              }
    # merge text
    text, start, end = derive_segment_text(merged)
    assert start == merged["starts"] and end == merged["ends"], "Mismatch in start / end of merged MC"
    merged["segment"] = text
    return merged


def handle_multiple_major_claims(major_claims, adus, resolve_ties="max_conf", apply_biases=("beginning", "merged"),
                                 bias_policy="at_least",
                                 discarded_handling="claim"):
    """

    Args:
        major_claims:
        adus:
        resolve_ties:
        apply_biases:
        bias_policy: How biases affect rejection / retaining of an MC: Can be:
            - "at_least": Keep MCs that satisfy at least one bias
            - "max": Keep MCs (generally single) that score the highest
            Remaining MCs will be resolved with tie breaking if > 1
        discarded_handling: What to do with discarded mcs. Can be:
            - "claim": Set them as regular claims
            - "drop": Discard them

    Returns:

    """
    if apply_biases is None:
        apply_biases = []
    slist = list(sorted(major_claims, key=lambda x: int(x["starts"])))
    merged_set = set()
    # first merge any contiguous mcs
    i = 0
    while i < len(slist) - 1:
        a = slist[i]
        b = slist[i + 1]
        if a['ends'] + 1 == b['starts']:
            # remove marginal MCs from the favored list
            merged_set.discard(a["id"])
            merged_set.discard(b["id"])
            # merge
            merged = merge_adu_segments(a, b)
            logging.info(
                f"Merging MCs: {a['id'], a['starts'], a['ends']} and {b['id'], b['starts'], b['ends']} to: {merged['segment']} ")
            # set merged as favored
            merged_set.add(merged["id"])
            slist = slist[:i] + [merged] + slist[i + 2:]
        else:
            i += 1

    major_claims = slist
    discarded_major_claims = []
    bias_scores = np.zeros(len(major_claims))
    for bias in apply_biases:
        # beginning bias
        if bias == "beginning":
            # favor an MC that starts at zero-ish
            bias_scores += np.asarray([s['starts'] < 5 for s in major_claims])
        elif bias == "merged":
            # favor merged MCs
            bias_scores += np.asarray([s['id'] in merged_set for s in major_claims])
    if bias_policy == "max":
        passing = bias_scores == bias_scores.max(axis=0)
    elif bias_policy == "at_least":
        passing = bias_scores > 0
    if np.any(passing):
        # if some bias-wise filtering suceeded, apply it; -- else proceed with tie breaking as is
        discarded_major_claims.extend(np.take(major_claims, np.where(~passing)[0]).tolist())
        major_claims = np.take(major_claims, np.where(passing)[0]).tolist()

    # tie breaks
    if len(major_claims) > 1:
        if resolve_ties == "max_conf":
            # keep only the mc that has the highest confidence
            mc_sorted = list(sorted(major_claims, key=lambda x: float(x['confidence'])))
            discarded, major_claims = mc_sorted[:-1], [mc_sorted[-1]]
            discarded_major_claims.extend(discarded)
        else:
            raise NotImplementedError(f"Undefined MC tiebreaker: {resolve_ties}")

    # handle discarded
    if discarded_handling == "claim":
        for dis in discarded_major_claims:
            dis["type"] = "claim"
            dis["label"] = "claim"
        adus.extend(discarded_major_claims)
    elif discarded_handling == "drop":
        pass
    else:
        raise NotImplementedError(f"Undefined discarded MC handling: {discarded_handling}")

    assert len(major_claims) == 1, f"More than one ({len(major_claims)}) MC after resolution!: {major_claims}"
    return adus, major_claims[0]


def handle_missing_major_claim(adus, first_sentence):
    toks_expanded, toks = first_sentence
    text = "".join(get_sentence_raw_tokens(toks_expanded))
    # set title as the major claim
    seg = {
        "id": make_segment_id(),
        "type": "major_claim",
        "label": "major_claim",
        "starts": toks_expanded[-2],
        "ends": str(toks_expanded[-1]),
        "segment": text,
        "tokens": toks,
        "confidences": [str(round(random.uniform(0.98, 0.999), 3)) for _ in toks],
        "expanded_tokens": toks,
        "source_sentences": [toks_expanded],
        "source_sentences_expanded": [toks],
        "confidence": str(round(random.uniform(0.98, 0.999), 3))
    }
    logging.info(f"Injecting document title: {text} as the fallback major claim.")
    # remove any adu with the same / overlapping segment content
    overlaps = [a for a in adus if a['segment'] == text or int(a['starts']) < int(seg['ends'])]
    if overlaps:
        logging.debug("Discarding ADUs for overlapping with inserted title major claim:")
        for o in overlaps:
            logging.debug(f"{o['starts'], o['ends']} - {o['segment']}")
    adus = [a for a in adus if a not in overlaps]
    return adus, seg
