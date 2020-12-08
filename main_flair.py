from flair_impl import preprocessing as prep
import pandas as pd
from os.path import join
from os import getcwd

if __name__ == '__main__':
    do_prep = True
    if do_prep:
        prep.preprocess()
    else:
        path_to_csv = join(getcwd(), "resources", "train.csv")
        df = pd.read_csv(path_to_csv, sep="\t", header=None, index_col=0)
        print(df)
