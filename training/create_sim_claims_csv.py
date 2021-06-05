import re
from os import getcwd
from os.path import join
from itertools import combinations

import pandas as pd
import numpy as np


def sim_data():
    path_to_data = join(getcwd(), "resources", "data")
    path_to_file = join(path_to_data, "train_adu.csv")
    data = pd.read_csv(path_to_file, sep="\t", index_col=None, header=None)
    df_list = np.split(data, data[data.isnull().all(1)].index)
    df_list = [df.dropna() for df in df_list]
    claims = []
    for df in df_list:
        sentence = list(df[0])
        labels = list(df[1])
        claim_tokens = []
        for token, label in zip(sentence, labels):
            if label == "B-claim" or label == "I-claim":
                claim_tokens.append(token)
        claim = " ".join(claim_tokens)
        re.sub(' +', ' ', claim)
        claims.append(claim)
    df = pd.DataFrame(columns=["claims"])
    claims = [claim for claim in claims if claim != "" and claim != " " and claim is not None]
    df["claims"] = claims
    file = "../output/test/claims.csv"
    df.dropna()
    df.to_csv(file, sep="\t", index=False, header=True)
    cluster_combinations = list(combinations(claims, r=2))
    df = pd.DataFrame(columns=["topic", "sentence_1", "sentence_2", "label"])
    row_counter = 0
    for pair_combination in cluster_combinations:
        sentence1 = pair_combination[0]
        sentence2 = pair_combination[1]
        if not sentence1 or not sentence2:
            continue
        print(f"creating pair for {sentence1} and {sentence2}")
        df.loc[row_counter] = ["", sentence1, sentence2, ""]
        row_counter += 1
    output_file = "train_sim_greek.tsv"
    df.to_csv(output_file, sep="\t", header=True, index=False)


def export_claim_pairs():
    initial_file = "../output/test/claims.csv"
    df = pd.read_csv(initial_file, sep="\t", header=0, index_col=None)
    claims = list(df["claims"])
    claim_pairs = list(combinations(claims, r=2))
    df = pd.DataFrame(columns=["claim1", "claim2", "label"])
    row_counter = 0
    for pair in claim_pairs:
        claim1 = pair[0]
        claim2 = pair[1]
        df.loc[row_counter] = [claim1, claim2, ""]
        row_counter += 1
    output_file = "sim_claims.tsv"
    df.to_csv(output_file, sep="\t", header=True, index=False)


if __name__ == '__main__':
    export_claim_pairs()
