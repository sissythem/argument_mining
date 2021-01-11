from os import getcwd
from os.path import join

import numpy as np
import pandas as pd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    path_to_resources = join(getcwd(), "resources")
    path_to_results = join(path_to_resources, "test.tsv")
    results = pd.read_csv(path_to_results, sep=" ", index_col=None, header=None, skip_blank_lines=False)
    df_list = np.split(results, results[results.isnull().all(1)].index)
    sentences = []
    for df in df_list:
        df = df[df[0].notna()]
        df[3] = np.where(df[1] == df[2], 0, 1)
        sentences.append(df)
    sentences_df = pd.concat(sentences)
    sentences_df.to_csv(join(path_to_resources, "results.csv"), sep="\t", index=None, header=None)
    error_sentences = []
    for sentence_df in sentences:
        if 1 in sentence_df[3].values:
            total_text = ""
            for index, row in sentence_df.iterrows():
                text, true_lbl, pred_lbl, diff = row
                total_text += f"{text} <{true_lbl}> " if diff == 0 else \
                    f"{text} <{true_lbl}> <{pred_lbl}> "
            print(total_text.strip())
            print("==============================================================================")
            error_sentences.append(total_text + "\n\n")
    with open(join(path_to_resources, "errors.txt"), "w") as f:
        f.writelines(error_sentences)


if __name__ == '__main__':
    main()
