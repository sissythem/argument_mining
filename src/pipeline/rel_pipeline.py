import logging
from src.utils import make_segment_id


def run(rel_model, inputs):
    """
    Relations prediction pipeline

    Args
        | major_claims (list): a list of the major claims predicted in the previous step
        | claims (list): a list of the claims predicted in the previous step
        | premises (list): a list of the premises predicted in the previous step

    Returns
        tuple: the list of the predicted relations and the counter used to produce ids for the relations
    """
    results = []
    for i, adus in enumerate(inputs):
        logging.info(f"Extracting relations for doc {i + 1}/{len(inputs)}")
        major_claims, claims, premises = adus["major_claims"], adus["claims"], adus["premises"]
        relations = []
        if major_claims and claims:
            relations += get_relations(rel_model, source=claims, target=major_claims)
        if claims and premises:
            relations += get_relations(rel_model, source=premises, target=claims, modify_source_list=True)
        results.append(relations)
    return results


def get_relations(rel_model, source, target, modify_source_list=False):
    """
    Performs combinations creating pairs of source/target ADUs to predict their relations. If the predicted label
    is either support or attack, the relations list is updated with a new entry.

    Args
        | source (list): list of source ADUs
        | target (list): list of target ADUs
        | counter (int): counter to be used in the new relations' ids

    Returns
        tuple: the list of the predicted relations and the counter with the new value
    """
    initial_source = source
    relations, already_predicted = [], []

    for adu2 in target:
        if modify_source_list:
            source = modify_source_adus(
                adu2, already_predicted, initial_source)
            if not source:
                break
        for adu1 in source:
            seg1, seg2 = adu1['segment'], adu2['segment']
            # sentence_pair = f"[CLS] {adu1['segment']} [SEP] {adu2['segment']}"
            logging.debug(f"Predicting relation for sentence pair: {(seg1, seg2)}")
            score, label = rel_model.predict(seg1, seg2)
            if label and label != rel_model.get_null_label():
                rel_dict = {
                    "id": make_segment_id(),
                    "type": label,
                    "arg1": adu1["id"],
                    "arg2": adu2["id"],
                    "confidence": str(score)
                }
                relations.append(rel_dict)
                already_predicted.append(adu1)
    return relations


def modify_source_adus(adu2, already_predicted, source):
    adu2_start = adu2["starts"]
    adu2_end = adu2["ends"]
    source = remove_already_predicted(
        source=source, already_predicted=already_predicted)
    if not source:
        return source
    source = keep_k_closest(
        source=source, target_start=adu2_start, target_end=adu2_end)
    return source


def remove_already_predicted(source, already_predicted):
    if source and already_predicted:
        final_source = []
        for s in source:
            found = False
            for pred in already_predicted:
                if pred["id"] == s["id"]:
                    found = True
                    break
            if not found:
                final_source.append(s)
        return final_source
    return source


def keep_k_closest(source, target_start, target_end, k=5):
    source = sorted(source, key=lambda key: int(
        key['starts']), reverse=False)
    for s in source:
        s["distance_from_start"] = abs(int(target_start) - int(s["ends"]))
        s["distance_from_end"] = abs(int(target_end) - int(s["starts"]))
    source_from_start = sorted(
        source, key=lambda key: key['distance_from_start'], reverse=False)
    if source_from_start:
        source_from_start = source_from_start[:k]
    source_from_end = sorted(
        source, key=lambda key: key['distance_from_end'], reverse=False)
    if source_from_end:
        source_from_end = source_from_end[:k]
    if source_from_start and source_from_end:
        combined_source = source_from_start + source_from_end
        final_source = []
        for source in combined_source:
            if final_source:
                found = False
                for s in final_source:
                    if s["id"] == source["id"]:
                        found = True
                if not found:
                    final_source.append(source)
            else:
                final_source.append(source)
    elif source_from_start and not source_from_end:
        final_source = source_from_start
    elif not source_from_start and source_from_end:
        final_source = source_from_end
    else:
        final_source = source
    return final_source
