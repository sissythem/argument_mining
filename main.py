from os.path import join

from elasticsearch_dsl import Search
from ellogon import esclient_swo

from arg_mining import AduModel, RelationsModel, ArgumentMining
from training_data import DataLoader
from utils import AppConfig


def collect_documents(app_config):
    client = esclient_swo.elastic_server_client
    file_path = join(app_config.resources_path, "kasteli_34_urls.txt")
    # read the list of urls from the file:
    with open(file_path, "r") as f:
        urls = [line.rstrip() for line in f]

    search_articles = Search(using=client, index='articles').filter('terms', link=urls)
    # print(search_articles.to_dict())
    found = 0
    for hit in search_articles.scan():
        document = hit.to_dict()
        arg_mining = ArgumentMining(app_config=app_config)
        arg_mining.predict(document=document)
        found += 1
    print(f"Found documents: {found}")
    esclient_swo.stop()


def main():
    app_config: AppConfig = AppConfig()
    logger = app_config.app_logger
    properties = app_config.properties
    do_prep = properties["do_prep"]
    if do_prep:
        data_loader = DataLoader(app_config=app_config)
        logger.info("Creating CSV file in CONLL format for ADUs classification")
        data_loader.load_adus()
        logger.info("Creating CSV file in CONLL format for relations classification")
        data_loader.load_relations()
        logger.info("Creating CSV file in CONLL format for stance classification")
    if "train" in properties["tasks"]:
        logger.info("Training ADU classifier")
        adu_model = AduModel(app_config=app_config)
        adu_model.train()
        logger.info("ADU Training is finished!")
        logger.info("Training relations model")
        rel_model = RelationsModel(app_config=app_config, dev_csv=app_config.rel_dev_csv,
                                   train_csv=app_config.rel_train_csv, test_csv=app_config.rel_test_csv,
                                   base_path=app_config.rel_base_path, model_name="rel")
        rel_model.train()
        logger.info("Relations training finished!")
        logger.info("Training stance model")
        stance_model = RelationsModel(app_config=app_config, dev_csv=app_config.stance_dev_csv, model_name="stance",
                                      train_csv=app_config.stance_train_csv, test_csv=app_config.stance_test_csv,
                                      base_path=app_config.stance_base_path)
        stance_model.train()
        logger.info("Stance training finished!")
    if "eval" in properties["tasks"]:
        collect_documents(app_config=app_config)
        logger.info("Evaluation is finished!")


if __name__ == '__main__':
    main()
