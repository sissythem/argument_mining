import pickle
from os.path import join

import pandas as pd


def adu_preprocess(app_config):
    resources = app_config.resources_path
    documents_path = join(resources, app_config.documents_pickle)
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)
    df = pd.DataFrame(columns=["token", "label", "arg", "doc_sentence", "sentence", "doc_id"])
    row_counter = 0
    sentence_counter = 0
    for document in documents:
        print("Processing document with id: {}".format(document.document_id))
        doc_sentence_counter = 0
        for idx, sentence in enumerate(document.sentences):
            print("Processing sentence: {}".format(sentence))
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
    print("Finished building dataframing. Saving...")
    out_file_path = join(resources, "train.csv")
    df.to_csv(out_file_path, sep='\t', index=False)
    print("Dataframe saved!")


def preprocess_relations():
    pass