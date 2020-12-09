import pickle
from os.path import join
from typing import List

import pandas as pd

from data.document_models import Document, Relation


def adu_preprocess(app_config):
    logger = app_config.app_logger
    logger.debug("Running ADU preprocessing")
    resources = app_config.resources_path
    documents_path = join(resources, app_config.documents_pickle)
    logger.debug("Loading documents from pickle file")
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)
    logger.debug("Documents are loaded")
    df = pd.DataFrame(columns=["token", "label", "is_arg", "sp", "sentence", "document"])
    row_counter = 0
    sentence_counter = 0
    for document in documents:
        logger.debug("Processing document with id: {}".format(document.document_id))
        doc_sentence_counter = 0
        for idx, sentence in enumerate(document.sentences):
            logger.debug("Processing sentence: {}".format(sentence))
            labels = document.sentences_labels[idx]
            for token, label in zip(sentence, labels):
                is_arg = "Y" if label != "O" else "N"
                sp = "SP: {}".format(doc_sentence_counter)
                sentence_counter_str = "Sentence: {}".format(sentence_counter)
                document_str = "Doc: {}".format(document.document_id)
                df.loc[row_counter] = [token, label, is_arg, sp, sentence_counter_str, document_str]
                row_counter += 1
                sentence_counter += 1
                doc_sentence_counter += 1
            df.loc[row_counter] = ["", "", "", "", "", ""]
            row_counter += 1
    logger.debug("Finished building dataframe. Saving...")
    out_file_path = join(resources, "train_adu.csv")
    df.to_csv(out_file_path, sep='\t', index=False, header=None)
    logger.debug("Dataframe saved!")


def preprocess_relations(app_config):
    relations, stances = _get_relations(app_config)
    _save_rel_df(logger=app_config.app_logger, rel_list=relations, resources_path=app_config.resources_path,
                 filename=app_config.rel_train_csv)
    _save_rel_df(logger=app_config.app_logger, rel_list=stances, resources_path=app_config.resources_path,
                 filename=app_config.stance_train_csv)


def _save_rel_df(logger, rel_list, resources_path, filename):
    df = pd.DataFrame(columns=["token", "label", "sentence"])
    row_counter = 0
    sentence_counter = 0
    for pair in rel_list:
        text1 = pair[0]
        text2 = pair[1]
        relation = pair[2]
        logger.debug("Processing pair:")
        logger.debug("Text 1: {}".format(text1))
        logger.debug("Text 2: {}".format(text2))
        logger.debug("Pair label: {}".format(relation))
        final_text = "[CLS] " + text1 + " [SEP] " + text2
        sentence_counter_str = "Pair: {}".format(sentence_counter)
        df.loc[row_counter] = [final_text, relation, sentence_counter_str]
        row_counter += 1
        sentence_counter += 1
    output_filepath = join(resources_path, filename)
    df.to_csv(output_filepath, sep='\t', index=False, header=None)
    logger.debug("Dataframe saved!")


def _collect_segments(documents: List[Document]):
    major_claims, claims, premises = {}, {}, {}
    relation_pairs, stance_pairs = {}, {}
    for document in documents:
        relations: List[Relation] = document.relations
        stances = document.stance
        for relation in relations:
            relation_pairs[(relation.arg1.segment_id, relation.arg2.segment_id)] = relation.relation_type
        for stance in stances:
            stance_pairs[(stance.arg1.segment_id, stance.arg2.segment_id)] = stance.relation_type
        for segment in document.segments:
            if segment.arg_type == "major_claim":
                major_claims[segment.segment_id] = segment.text
            elif segment.arg_type == "claim":
                claims[segment.segment_id] = segment.text
            elif segment.arg_type == "premise":
                premises[segment.segment_id] = segment.text
            else:
                continue
    return major_claims, claims, premises, relation_pairs, stance_pairs


def _collect_relation_pairs(parents, children, relation_pairs):
    new_relation_pairs = []
    count_relations = 0
    for p_id, p_text in parents.items():
        for c_id, c_text in children.items():
            key = (c_id, p_id)
            if key in relation_pairs.keys():
                print("Found relation")
                count_relations += 1
            relation = relation_pairs.get(key, "other")
            new_relation_pairs.append((c_text, p_text, relation))
    print("Found {} relations".format(count_relations))
    return new_relation_pairs


def _get_relations(app_config):
    logger = app_config.app_logger
    resources = app_config.resources_path
    documents_path = join(resources, app_config.documents_pickle)
    logger.debug("Loading documents from pickle file")
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)
    logger.debug("Documents are loaded")
    major_claims, claims, premises, relation_pairs, stance_pairs = _collect_segments(documents)
    relations = _collect_relation_pairs(parents=major_claims, children=claims, relation_pairs=relation_pairs)
    relations += _collect_relation_pairs(parents=claims, children=premises, relation_pairs=relation_pairs)
    stances = _collect_relation_pairs(parents=major_claims, children=claims, relation_pairs=stance_pairs)
    return relations, stances
