import traceback

from src.pipeline.debatelab import DebateLab
from src.training.models import SequentialModel, ClassificationModel
from src.training.preprocessing import DataPreprocessor
from src.training_hf.hf_datasets import *
from src.training_hf.models import *
from src.utils.config import Notification
from baseline import run_baseline

from sklearn.dummy import DummyClassifier
import numpy as np


def error_analysis(path_to_resources):
    """
    Function to perform error analysis on the results. Saves the incorrect predictions into a file

    Args
        path_to_resources (str): the full path to the resources folder
    """
    path_to_results = join(path_to_resources, "resources/results", "test.tsv")
    results = pd.read_csv(path_to_results, sep=" ", index_col=None, header=None, skip_blank_lines=False)
    df_list = np.split(results, results[results.isnull().all(1)].index)
    sentences = []
    for df in df_list:
        df = df[df[0].notna()]
        df[3] = np.where(df[1] == df[2], 0, 1)
        sentences.append(df)
    sentences_df = pd.concat(sentences)
    sentences_df.to_csv(join(path_to_resources, "resources/results/results.tsv"), sep="\t", index=False, header=False)
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


def train(app_config):
    """
    Train the selected models. In the application properties, the models to be trained are indicated.

    Args
        app_config (AppConfig): the application configuration
    """
    logger = app_config.app_logger
    models_to_train = app_config.properties["train"]["models"]
    if "adu" in models_to_train:
        logger.info("Training ADU classifier")
        adu_model = SequentialModel(app_config=app_config, model_name="adu")
        adu_model.train()
        logger.info("ADU Training is finished!")
    if "rel" in models_to_train:
        logger.info("Training relations model")
        rel_model = ClassificationModel(app_config=app_config, model_name="rel")
        rel_model.train()
        logger.info("Relations training finished!")
    if "stance" in models_to_train:
        logger.info("Training stance model")
        stance_model = ClassificationModel(app_config=app_config, model_name="stance")
        stance_model.train()
        logger.info("Stance training finished!")
    if "sim" in models_to_train:
        logger.info("Training argument similarity model")
        sim_model = ClassificationModel(app_config=app_config, model_name="sim")
        sim_model.train()
        logger.info("Finished training similarity model")


def main():
    """
    The main function of the program. Initializes the AppConfig class to load the application properties and
    configurations and based on the tasks in the properties executes the necessary steps (preprocessing, training,
    DebateLab pipeline, error analysis)
    """
    app_config: AppConfig = AppConfig()
    notification: Notification = Notification(app_config=app_config)
    try:
        properties = app_config.properties
        tasks = properties["tasks"]
        if "prep" in tasks:
            data_preprocessor = DataPreprocessor(app_config=app_config)
            data_preprocessor.preprocess()
        if "train" in tasks:
            train(app_config=app_config)
        if "eval" in properties["tasks"]:
            arg_mining = DebateLab(app_config=app_config)
            arg_mining.run_pipeline()
        if "error" in properties["tasks"]:
            error_analysis(path_to_resources=app_config.resources_path)
        notification.send_email(body="Argument mining pipeline finished successfully",
                                subject=f"Argument mining run: {app_config.run}")
    except(BaseException, Exception):
        app_config.app_logger.error(traceback.format_exc())
        notification.send_email(
            body=f"Argument mining pipeline finished with errors {traceback.format_exc(limit=100)}",
            subject=f"Error in argument mining run: {app_config.run}")
    finally:
        try:
            app_config.elastic_save.stop()
            app_config.elastic_retrieve.stop()
        except(BaseException, Exception):
            app_config.app_logger.error("Could not close ssh tunnels")
            exit(-1)


def main_huggingface():
    app_config = AppConfig()
    arg_mining_dataset = ArgMiningDataset(app_config=app_config)
    arg_mining_model = TransformerModel(app_config=app_config)
    model_id = "bert-base-uncased"
    seqlen=16
    # tok, num_labels, train_dset, eval_dset = arg_mining_dataset.load_data(model_id="xlm-roberta-base", seqlen=512, limit_data=100)
    tok, num_labels, train_dset, eval_dset = arg_mining_dataset.load_data(model_id=model_id, seqlen=seqlen, limit_data=100)
    # arg_mining_model.train(model_id=model_id, tokenizer=tok, num_labels=num_labels, train_dset=train_dset, eval_dset=eval_dset, seqlen=seqlen, batch_size=8, eval_step_period=100, lr=0.01, epochs=1000)
    
   
    # baseline code
    ################################3
    
    train_dat = torch.stack([t['input_ids'] for t in train_dset])
    train_dat = [tok.convert_ids_to_tokens(t) for t in  train_dat]
    train_lab = np.stack([np.asarray(t['labels']) for t in train_dset])
    train_lab[train_lab == -100] = 7

    eval_dat = torch.stack([t['input_ids'] for t in eval_dset])
    eval_dat = [tok.convert_ids_to_tokens(t) for t in  eval_dat]
    eval_lab = np.stack([np.asarray(t['labels']) for t in eval_dset])
    eval_lab[eval_lab == -100] = 7

    print("Training with baseline")
    model, w2i, t2i = run_baseline(train_dat, train_lab)
    print("Evaluating with baseline")
    run_baseline(eval_dat, eval_lab, model=(model, w2i, t2i))

if __name__ == '__main__':
    main()
    # main_huggingface()
    print("Done!")
