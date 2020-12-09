# from flair.data import TaggedCorpus
# from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
import csv
import glob
import os
import pickle
from pathlib import Path
from os import getcwd
from os.path import join
# sudo apt-get install python3-tk
from tkinter import Tcl
from typing import List, Any

import pandas as pd
import numpy as np
# https://github.com/facebook/Ax
# pip3 install --user ax-platform
from ax import optimize
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, BertEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentEmbeddings, \
    WordEmbeddings, PooledFlairEmbeddings, TransformerWordEmbeddings
from sklearn.metrics import classification_report
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
import torch


def experiment(
        corpus,
        embeddings,
        tag_dictionary,
        tag_type,
        hidden_size=256,
        rnn_layers=2,
        use_crf=True,
        optimizer=Adam,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=150,
        train_with_dev=False,
        path='resources/taggers/argmin',
        save_final_model=False,
        num_workers=8,
        seed=2019,
        embedding_names="",
        embedding_storage_mode='cpu'
):
    # Set seed...
    import random
    random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)

    # Create a base path:
    base_path = path + '-' + '-'.join([
        str(embedding_names),
        'hs=' + str(hidden_size),
        'hl=' + str(rnn_layers),
        'crf=' + str(use_crf),
        str(optimizer.__name__),
        'lr=' + str(learning_rate),
        'bs=' + str(mini_batch_size),
        's=' + str(seed)
    ])
    try:
        # os.mkdir(base_path, 0o755)
        os.makedirs(base_path)
    except (OSError, Exception):
        pass

    if not os.path.isfile(base_path + '/weights.txt'):
        with open(base_path + '/embeddings-info.txt', 'w') as fd:
            print(embeddings, file=fd)

        from flair.models import SequenceTagger
        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=use_crf,
                                                rnn_layers=rnn_layers)

        # 6. initialize trainer
        from flair.trainers import ModelTrainer

        # Params:
        #  model
        #  corpus
        #  optimizer = SGD
        #  epoch = 0
        #  loss = 10000.0
        #  optimizer_state = None
        #  scheduler_state = None
        trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=optimizer)

        # 7. start training
        # Params:
        #  base_path
        #  -- evaluation_metric = EvaluationMetric.MICRO_F1_SCORE
        #  learning_rate = 0.1
        #  mini_batch_size = 32
        #  eval_mini_batch_size = None (becomes mini_batch_size)
        #  max_epochs = 100
        #  anneal_factor = 0.5
        #  patience = 3
        #  min_learning_rate: float = 0.0001
        #  train_with_dev = False
        #  monitor_train = False
        #  monitor_test = False
        #  embedding_storage_mode = "cpu"
        #  -- embeddings_in_memory = True
        #  checkpoint = False
        #  save_final_model = True
        #  anneal_with_restarts = False
        #  shuffle = True
        #  param_selection_mode = False
        #  num_workers = 6
        #  sampler = None
        trainer.train(base_path,
                      learning_rate=learning_rate,
                      mini_batch_size=mini_batch_size,
                      max_epochs=max_epochs,
                      train_with_dev=train_with_dev,
                      save_final_model=save_final_model,
                      num_workers=num_workers,
                      shuffle=False,  ## TODO remove
                      # embedding_storage_mode=embedding_storage_mode,
                      monitor_test=True)

        # 8. plot training curves (optional)
        from flair.visual.training_curves import Plotter
        plotter = Plotter()
        plotter.plot_training_curves(base_path + '/loss.tsv')
        try:
            plotter.plot_weights(base_path + '/weights.txt')
        except:
            open(base_path + '/weights.txt', 'a').close()

    if not os.path.isfile(base_path + '/evaluation_report.txt'):
        with open(base_path + '/evaluation_report.txt', 'w') as fd:
            # 9. evaluate and return best macro F1.
            names = ['Word', 'Gold', 'Predicted', 'Prob']
            fnames = glob.glob(base_path + '/test-epoch-*.tsv')
            fnames = Tcl().call('lsort', '-dictionary', fnames)
            best = 0
            best_epoch = 0
            for fname in fnames:
                filename = os.path.basename(fname)
                epoch = int(filename[11:-4])
                epoch += 1
                df = pd.read_csv(fname, encoding="utf-8", sep=' ',
                                 names=names, quoting=csv.QUOTE_NONE).fillna(method="ffill")
                tags_vals = list(set(df["Gold"].values))
                tags_vals.sort()
                gold = df["Gold"].values
                predicted = df["Predicted"].values
                report = classification_report(gold, predicted, digits=4, output_dict=True)
                print('Epoch:', '%3d' % epoch,
                      ', accuracy:', '%.2f' % (100. * report['accuracy']),
                      ', macro F1:', '%.2f' % (100. * report['macro avg']['f1-score']),
                      ', macro P:', '%.2f' % (100. * report['macro avg']['precision']),
                      ', macro R:', '%.2f' % (100. * report['macro avg']['recall']),
                      file=fd)
                f1 = report['macro avg']['f1-score']
                if (f1 > best):
                    best = f1
                    best_epoch = epoch
            print("Best epoch:", best_epoch, ', F1:', best, file=fd)
        print("Best epoch:", best_epoch, ', F1:', best)
        with open(base_path + '/evaluation_report_best.bin', 'wb') as fd:
            pickle.dump(best, fd)
    else:
        with open(base_path + '/evaluation_report_best.bin', 'rb') as fd:
            best = pickle.load(fd)
    return best


def hyperparams_tunning(corpus, embeddings, embedding_names, tag_dictionary, tag_type, params,
                        path='resources/taggers/argmin-tune', save_final_model=False, num_workers=8):
    score = experiment(corpus=corpus, embeddings=embeddings, embedding_names=embedding_names,
                       tag_dictionary=tag_dictionary, tag_type=tag_type,
                       hidden_size=params.get('hidden_size', 256),
                       rnn_layers=params.get('rnn_layers', 2),
                       use_crf=params.get('use_crf', True),
                       optimizer=params.get('optimizer', Adam),
                       learning_rate=params.get('learning_rate', 0.1),
                       mini_batch_size=params.get('mini_batch_size', 32),
                       max_epochs=params.get('max_epochs', 150),
                       seed=params.get('seed', 2019),
                       embedding_storage_mode=params.get('embedding_storage_mode', 'cpu'),
                       path=path, save_final_model=save_final_model, num_workers=num_workers
                       )
    print('score:', score, params)
    return score


# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
# data_folder = '/home/petasis/DeepLearning/ArgumentMining/datasets/ArgumentAnnotatedEssays-2.0/dataset_conll/nodev'
curr_dir = Path(getcwd())
curr_dir = str(curr_dir) if str(curr_dir).endswith("argument_mining") else str(curr_dir.parent)
data_folder = join(curr_dir, "resources")

# retrieve corpus using column format, data folder and the names of the train, dev and test files
# corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
#                                                               train_file='train.txt',
#                                                               test_file='test.txt',
#                                                               dev_file='dev.txt')
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train_adu.csv',
                              test_file='train_adu.csv',
                              dev_file='train_adu.csv')
print(len(corpus.train))
print(corpus.train[4].to_tagged_string('pos'))
print(corpus.train[4].to_tagged_string('ner'))

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
    # PooledFlairEmbeddings(),
    # ELMoEmbeddings(),
    # ELMoTransformerEmbeddings(),
    # TransformerXLEmbeddings(),
    # OpenAIGPTEmbeddings(),
    # Models in /home/petasis/.local/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling.py
    # BertEmbeddings('bert-large-uncased'),
    TransformerWordEmbeddings('nlpaueb/bert-base-greek-uncased-v1'),
    # SiameseEmbeddings(sentence_folder),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embedding_types)


# embeddings: GenericStackedEmbeddings = GenericStackedEmbeddings(embedding_types)


# Run experiment...
# experiment(corpus=corpus, embeddings=embeddings, embedding_names='glove-flair-news-fb-bert-base-uncased',
#            tag_dictionary=tag_dictionary, tag_type=tag_type)

def optimize_callback(p):
    # print(p)
    p['max_epochs'] = 150
    p['seed'] = 2019
    # p['hidden_size']            = 256
    # p['rnn_layers']             = 2
    # p['embedding_storage_mode'] = 'gpu'
    # p['learning_rate']   = 0.2
    embedding_names = 'glove-flairnewsfb-bertbaseuncased'
    embedding_names = 'bert-large'
    return hyperparams_tunning(corpus, embeddings, embedding_names, tag_dictionary, tag_type, p,
                               # path='resources/taggers/argmin-tune')
                               path='resources-seed-2019-bert-large/taggers/argmin-tune')
    # try:
    # except:
    #    return 0


P = dict()
P['hidden_size'] = {
    'name': 'hidden_size',
    'type': 'choice',
    'values': [64, 128, 256, 512, 1024]
}
P['rnn_layers'] = {
    'name': 'rnn_layers',
    'type': 'range',
    'bounds': [1, 4]
}
# import random
# random.seed(2019)
# print(2019)
# for x in range(9):
#   print(random.randint(1,1000000000))
P['seed'] = {
    'name': 'seed',
    'type': 'choice',
    'values': [2019, 893677899, 165985916, 847106131, 260119319, 533767820,
               172439050, 263200773, 696701653, 267742788]
}
params = [P['hidden_size'], P['rnn_layers']]
# params = [P['seed']]
best_parameters, best_values, experiment, model = optimize(
    parameters=params,
    evaluation_function=optimize_callback,
    minimize=False,
    objective_name='Macro F1',
    total_trials=5
)
# embedding_names = 'bert-large'
# p = {"hidden_size": 256, "rnn_layers": 4, "seed": 2019}
# score = hyperparams_tunning(corpus, embeddings, embedding_names, tag_dictionary, tag_type, p,
#                             path='resources-seed-2019-bert-large/taggers/argmin-tune')
# print(score)
print('Best params:', best_parameters)
print('Best values:', best_values)
