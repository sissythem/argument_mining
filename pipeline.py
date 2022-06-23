"""Execute the entire debatelab pipeline"""

import argparse
import os
from os.path import join
import json
from threading import Lock

from src.annotation import identify_promising_docs_for_annotation, save_promising_docs_for_annotation
from src.adu_models import ADUModel
from src.embedder_model import Embedder
from src.utils import timestamp, skip_content_insertion, suspend_logging_verbosity, strip_document_extra_info

from src.pair_classifier_models import RelationClassifier, StanceClassifier
from src.io.elastic_io import ElasticSearchConfig
from src.utils import normalize_newlines, read_json_str_or_file, make_segment_id, MODEL_VERSION
from src.pipeline import adu_pipeline, rel_pipeline, stance_pipeline
from src.pipeline.entities import get_named_entities
from src.pipeline.topics import TopicModel
from src.pipeline.validation import validate_document_collection
from src.notification import Notification
from src.pipeline.clustering import run_clustering
from src.pipeline.crossdoc import write_additional_cross_doc_inputs, read_additional_cross_doc_inputs, \
    post_additional_cross_doc_inputs
from src.document import Document

import time
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_ADDITIONAL_CROSSDOCS_PATH = "additional_cross_document_inputs.json"


def get_args(input_args=None):
    """Argument definition function"""

    parser = argparse.ArgumentParser()
    # high-level
    parser.add_argument("-run_id", help="Number of gpus", required=False)
    parser.add_argument("-n_gpu", help="Number of gpus", default=1)
    # models
    parser.add_argument("-adu_model", help="ADU model", default="data/adu")
    parser.add_argument("-rel_model", help="REL model", default="data/rel")
    parser.add_argument("-stance_model", help="STANCE model", default="data/stance")
    parser.add_argument("-embedder_model", help="Generic text embedding model",
                        default="nlpaueb/bert-base-greek-uncased-v1")

    parser.add_argument("-device", help="Compute device", default="cpu")

    # i/o
    parser.add_argument("-input_type", help="Type of input", choices=["elastic", "instance", "output"])
    parser.add_argument("-input_path", help="Input path for text instance or existing outputs.", required=False)
    parser.add_argument("-input_title", help="Input title instance", required=False)

    parser.add_argument("-elastic_config", help="Json with elastic configuration")

    parser.add_argument("--save_to_elastic", help="Whether to save results to elastic.", default=False,
                        action="store_true")
    parser.add_argument("--disable_save_outputs", help="Switch to disable output saving", default=False,
                        action='store_true')
    parser.add_argument("--newline_norm", help="Wether to perform newline normalization", default=False,
                        action="store_true")
    parser.add_argument("--overwrite_topics",
                        help="Whether to overwrite existing topic extr. results (only relevant when 'inputs' reads existing outputs).",
                        default=False,
                        action="store_true")
    parser.add_argument("--overwrite_ner",
                        help="Whether to overwrite existing NER results (only relevant when 'inputs' reads existing outputs).",
                        default=False,
                        action="store_true")

    # metadata
    parser.add_argument("-ner_config", help="NER extraction configuration",
                        default="""{"endpoint": "http://petasis.dyndns.org:7100/predict-ner"}""")
    parser.add_argument("-topics_config", help="Topic extraction configuration",
                        default="""{"n_clusters": 10, "model_name": "nlpaueb/bert-base-greek-uncased-v1"}""")

    parser.add_argument("-notify_ics_credentials", help="Path to credentials file, to use to notify ICS")
    parser.add_argument("--export_schema", help="Whether to export the json schema", action="store_true", default=False)

    # cross-doc
    parser.add_argument("--use_additional_cross_doc_inputs", action="store_true", default=False)
    parser.add_argument("-cross_doc_clustering_config", help="Cross-document relation clustering configuration",
                        default="""{"n_clusters":16, "model_name": "nlpaueb/bert-base-greek-uncased-v1", "clustering": "similarity", "threshold": 0.9}""")
    parser.add_argument("--run_cross_doc_clustering", help="Switch for cross-document relation clustering",
                        action="store_true")

    parser.add_argument("-additional_cross_doc_inputs_path",
                        help="Path to additional inputs for cross-document clustering",
                        default=DEFAULT_ADDITIONAL_CROSSDOCS_PATH)
    parser.add_argument("-write_output_type_for_future_cross_doc",
                        help="Where to write outputs for future cross-document clustering consideration",
                        default=None, choices=["disk", "url"])

    # switches
    parser.add_argument("--run_topics", help="Topic extraction switch", default=False, action="store_true")
    parser.add_argument("--run_ner", help="Topic extraction switch", default=False, action="store_true")
    # adu must always run
    parser.add_argument("--disable_rel", help="Relation extraction switch", default=False, action="store_true")
    parser.add_argument("--disable_stance", help="Stance extraction switch", default=False, action="store_true")
    parser.add_argument("--disable_validation", help="Topic extraction switch", default=False, action="store_true")
    parser.add_argument("--strip_document_extra_info", help="Switch for removing extraneous doc processing information",
                        default=False, action="store_true")

    args = parser.parse_args()
    return args


def setup_logging(will_save_outputs=True, logging_folder="pipeline_outputs", logging_filename_prefix="logs"):
    """
    Logging configuration
    Args:
        logging_folder: Path to log file storage

    Returns:
        The logger object
    """
    logger = logging.getLogger()
    logger.setLevel(min(logger.level, logging.INFO))
    logging.getLogger('paramiko').setLevel(logging.ERROR)
    log_filename = f"{logging_filename_prefix}_{timestamp()}.log"

    log_formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s')
    os.makedirs(logging_folder, exist_ok=True)
    logfile_path = f"{logging_folder}/{log_filename}"
    file_handler = RotatingFileHandler(logfile_path, maxBytes=1024 * 1024 * 5, backupCount=10)
    file_handler.setFormatter(log_formatter)

    # yeet all handlers
    [logger.removeHandler(h) for h in logger.handlers]
    # add file handler
    logger.addHandler(file_handler)

    # add console output only if output is to be saved; else stdout is reserved for outputs
    if will_save_outputs:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    else:
        # if we will not save outputs, they will be flushed to stdout
        # suspend all logging
        suspend_logging_verbosity()
    return logfile_path


class Pipeline:
    def __init__(self, config_dict, persistent_run=False):

        self.reset_models()
        self.models_dict = {}
        self.model_paths_dict = {}
        self.config_dict = {}
        self.config = self.update_configuration(config_dict)
        # elasticsearch saving endpoint
        self.es_save = None
        self.es_retrieve = None
        self.persistent_run = persistent_run

        self.logfile_path = setup_logging(will_save_outputs=not self.config.disable_save_outputs,
                                          logging_filename_prefix=f"logs_{self.config.run_id}")

        # lock for pipeline object-based additional crossdoc IO
        self.additional_crossdocs_lock = Lock()
        self.additional_crossdocs = []

        # lock for models IO
        self.models_lock = Lock()
        self.in_use_lock = Lock()
        # generic lock
        self.lock = Lock()

    def set_in_use(self):
        self.lock.acquire()

    def release_use(self):
        self.lock.release()

    def is_in_use(self):
        return self.lock.locked()

    async def insert_crossdocs(self, docs):
        """
        Insert documents for future cross-document clustering. Filter for only new documents by ID
        """
        async with self.additional_crossdocs_lock:
            ids = [doc["id"] for doc in self.additional_crossdocs]
            self.additional_crossdocs += [doc for doc in docs if doc["id"] not in ids]
            time.sleep(10)

    async def remove_crossdocs(self, docs):
        """
        Remove documents for future cross-document clustering. Filter for only new documents by ID
        """
        async with self.additional_crossdocs_lock:
            self.additional_crossdocs = [x for x in self.additional_crossdocs if x not in docs]

    def reset_models(self):
        self.adu_model, self.rel_model, self.stance_model, self.embedder_model = None, None, None, None

    def update_configuration(self, config_dict):
        run_id = config_dict.get("run_id", "run_" + timestamp())
        if run_id is None:
            run_id = "run_" + timestamp()
        config_dict["run_id"] = run_id
        config_dict["export_schema"] = config_dict.get("export_schema", False)
        config_dict["input_title"] = config_dict.get("input_title", None)
        config_dict["disable_save_outputs"] = config_dict.get("disable_save_outputs", False)
        config_dict["notify_ics_credentials"] = config_dict.get("notify_ics_credentials", None)
        config_dict["overwrite_topics"] = config_dict.get("overwrite_topics", False)
        config_dict["overwrite_ner"] = config_dict.get("overwrite_ner", False)
        config_dict["save_to_elastic"] = config_dict.get("save_to_elastic", False)

        config_dict["disable_validation"] = config_dict.get("disable_validation", False)
        config_dict["disable_rel"] = config_dict.get("disable_rel", False)
        config_dict["disable_stance"] = config_dict.get("disable_stance", False)
        config_dict["strip_document_extra_info"] = config_dict.get("strip_document_extra_info", False)

        config_dict["use_additional_cross_doc_inputs"] = config_dict.get("use_additional_cross_doc_inputs", False)
        config_dict["additional_cross_doc_inputs_path"] = config_dict.get("additional_cross_doc_inputs_path",
                                                                          DEFAULT_ADDITIONAL_CROSSDOCS_PATH)
        config_dict["write_output_type_for_future_cross_doc"] = config_dict.get(
            "write_output_type_for_future_cross_doc", None)

        config = argparse.Namespace(**config_dict)
        return config

    def models_loaded(self):
        return self.adu_model is not None

    def load_store_models(self, models_config, device="cpu", store=False):
        """
        Function that loads all required resources for pipeline run
        Args:
            adu_model: Path / ID for ADU model
            rel_model: Path / ID for REL  model
            stance_model: Path / ID for STANCE model
            embedding_model_path: Path / ID for embedding model
            device: device to run on

        Returns:
            tuple of all loaded models
        """
        # engage lock
        with self.lock:
            for model_type, model_path in models_config.items():
                model = self.load_model(model_path, model_type, device, store=store)
                if model is None:
                    raise ValueError(f"Unable to load model: {model_type, model_path}")

    def load_model(self, model_path, model_type, device="cuda", store=False):
        """
        Function to load a single model in the pipeline
        Args:
            model_path: Name / path to model
            model_type: Model type -- can be: adu, rel, stance, embedder
            device: pytorch device id

        Returns:

        """
        model_class_mapping = {
            "adu": ADUModel, "rel": RelationClassifier, "stance": StanceClassifier, "embedder": Embedder
        }
        # check already loaded
        try:
            if self.model_paths_dict[model_type] == model_path and self.models_dict is not None:
                logging.info(f"Model {model_type} - {model_path} already loaded!")
                return self.models_dict[model_type]
        except KeyError:
            pass
        # instantiate and return
        logging.info(f"Reading {model_type} model from {model_path}")
        model = model_class_mapping[model_type](model_path, device=device)
        if store:
            if model_type == "adu":
                self.adu_model = model
            elif model_type == "rel":
                self.rel_model = model
            elif model_type == "stance":
                self.stance_model = model
            elif model_type == "embedder":
                self.embedder_model = model
            else:
                raise ValueError(f"Encountered undefined model type: {model_type}")
            self.models_dict[model_type] = model
            self.model_paths_dict[model_type] = model_path
        return model

    def run(self, config=None):
        """
        Pipeline running function
        """
        if config is None:
            config = self.config
        # read elastic configuration
        logging.info("Reading elastic config")
        with open(config.elastic_config) as f:
            elastic_config = json.load(f)

        raw_documents, pipeline_already_ran = self.get_inputs(input_type=config.input_type,
                                                              input_path=config.input_path,
                                                              input_title=config.input_title,
                                                              elastic_config=elastic_config["retrieve"],
                                                              newline_norm=config.newline_norm)
        # assign run_id
        for doc in raw_documents:
            doc["run_id"] = self.config.run_id

        # check notification credentials are ok, if supplied
        if config.notify_ics_credentials:
            config.notify_ics_credentials = read_json_str_or_file(config.notify_ics_credentials)
        # load trained models
        if not pipeline_already_ran:
            if not self.models_loaded():
                models_config = {
                    "adu": config.adu_model,
                    "rel": config.rel_model,
                    "stance": config.stance_model,
                    "embedder": config.embedder_model
                }
                self.load_store_models(models_config, store=True)

            # run ADU / REL / STANCE pipeline
            documents = Pipeline.run_pipeline(raw_documents, self.adu_model, self.rel_model, self.stance_model,
                                              cleanup_memory=not self.persistent_run,
                                              run_rel=not config.disable_rel,
                                              run_stance=not config.disable_stance)

        else:
            documents = raw_documents

        # run entity and topic metadata extraction
        documents = Pipeline.extract_metadata(documents, self.embedder_model, config.topics_config,
                                              config.ner_config,
                                              config.run_topics,
                                              config.run_ner,
                                              config.overwrite_topics, config.overwrite_ner)

        # get counters
        adu_counts, rel_counts, stance_counts = [], [], []
        for doc in documents:
            adus = doc['annotations']['ADUs']
            rels = doc['annotations']['Relations']
            claims = [c for c in adus if c['type'] == 'claim']
            adu_ids = [c['id'] for c in adus]
            if not config.disable_stance:
                stance_ids = [c['stance'][0]['id'] for c in claims]
                stance_counts.append(len(stance_ids))
            rel_ids = [c['id'] for c in rels]
            adu_counts.append(len(adu_ids))
            rel_counts.append(len(rel_ids))
        counters = {"adu": adu_counts, "rel": rel_counts, "stance": stance_counts}

        # run validation
        if config.save_to_elastic:
            if self.es_save is None:
                self.es_save = ElasticSearchConfig(elastic_config["save"])
        if not config.disable_validation:
            valid_ids = validate_document_collection(documents, counters,
                                                     elastic_save=self.es_save,
                                                     export_schema=config.export_schema)
            logging.info(f"Total of {len(valid_ids)} valid document ids: {valid_ids}")
        else:
            valid_ids = [doc["id"] for doc in documents]

        # save to elastic
        if self.es_save is not None:
            logging.info(f"Saving {len(valid_ids)} valid mining results to ES")
            for doc in documents:
                if doc["id"] in valid_ids:
                    self.es_save.save_document(document=doc)

        valid_docs = [doc for doc in documents if doc["id"] in valid_ids]

        # notify ICS on documents
        if config.notify_ics_credentials:
            n = Notification(config.notify_ics_credentials)
            n.notify_ics(ids_list=valid_ids)

        # perform cross-document clustering
        cross_doc_relations = []
        if config.run_cross_doc_clustering:
            ran_crossdoc = False
            cross_doc_inputs = valid_docs
            # read additional inputs
            if config.use_additional_cross_doc_inputs and os.path.exists(config.additional_cross_doc_inputs_path):
                # read and add new samples
                existing_docs = read_additional_cross_doc_inputs(config.additional_cross_doc_inputs_path)
                new_docs = [x for x in existing_docs if x not in [doc["id"] for doc in cross_doc_inputs]]
                if new_docs:
                    logging.info(f"Loaded {len(new_docs)} additional documents for crossdoc clustering from disk")
                cross_doc_inputs = cross_doc_inputs + new_docs

            if len(cross_doc_inputs) > 1:
                cross_doc_relations = Pipeline.run_cross_document_clustering(config, cross_doc_inputs,
                                                                             self.embedder_model,
                                                                             config.cross_doc_clustering_config,
                                                                             es_save=self.es_save,
                                                                             output_file="cross_doc_relations_" + config.run_id + ".json",
                                                                             notify_ics_creds=config.notify_ics_credentials)
                ran_crossdoc = True
            else:
                logging.info(f"Skipping document clustering for {len(valid_docs)} valid documents")

            if not ran_crossdoc and config.use_additional_cross_doc_inputs:
                # if the crossdoc procedure didn't run, restore the loaded docs back to the accumulation
                path, crossdoc = config.additional_cross_doc_inputs_path, config.write_output_type_for_future_cross_doc
                if crossdoc == "disk":
                    # write current documents to be considered for future cross-document clustering runs
                    write_additional_cross_doc_inputs(path, documents=new_docs)
                elif crossdoc == "url":
                    post_additional_cross_doc_inputs(path, documents=new_docs)
                else:
                    raise ValueError(f"Undefined future cross doc endpoint: {crossdoc}")

        # identify promising docs
        promising = identify_promising_docs_for_annotation(documents)
        # save / return / print overall outputs
        output_folder = "pipeline_outputs"
        outputs = Pipeline.prepare_outputs(config, documents, valid_ids)
        outputs["logfile"] = self.logfile_path
        if config.disable_save_outputs:
            print(json.dumps(outputs, ensure_ascii=False))
        else:
            output_file = config.run_id + ".json"
            os.makedirs(output_folder, exist_ok=True)
            with open(join(output_folder, output_file), "w", encoding="utf-8") as f:
                logging.info(f"Writing pipeline outputs to {f.name}")
                json.dump(outputs, f, ensure_ascii=False)

        save_promising_docs_for_annotation(promising)
        with open(join(output_folder, f"promising_{config.run_id}.json"), "w") as f:
            json.dump(promising, f)

        self.cleanup()
        logging.info(f"Pipeline done -- logs are at: {self.logfile_path}")
        return {"documents": documents, "cross_doc_relations": cross_doc_relations}

    @staticmethod
    def run_cross_document_clustering(arguments, documents, embedder, config, output_folder="cross_doc_relations",
                                      es_save=None, notify_ics_creds=None, output_file="cross_document_relations.json"):

        logging.info(f"Running cross-document clustering on {len(documents)} documents")
        config_ = read_json_str_or_file(config)
        relations = run_clustering(documents=documents, config=config_, embedder=embedder)
        logging.info(f"Discovered {len(relations)} cross-doc relations.")
        # ensure no intra-doc relations exist
        for rel in relations:
            assert rel['doc1_id'] != rel[
                'doc2_id'], "Found reflective (intra) cross-document relation!"

        # save to ES
        if es_save is not None:
            for relation in relations:
                es_save.save_relation(relation=relation)
        # save to disk
        os.makedirs(output_folder, exist_ok=True)
        with open(join(output_folder, output_file), "w", encoding="utf-8") as f:
            logging.info(f"Writing cross-doc relations to {f.name}")
            for rel in relations:
                rel['score'] = str(rel['score'])
            json.dump({"crossdoc_relations": relations, "arguments": vars(arguments)}, f, ensure_ascii=False)

        # notify ICS
        if notify_ics_creds:
            n = Notification(notify_ics_creds)
            n.notify_ics(ids_list=[r["id"] for r in relations], kind="clustering")
        return relations

    @staticmethod
    def extract_metadata(documents, embedder, topics_config, ner_config, run_topics=True, run_ner=True,
                         overwrite_topics=False,
                         overwrite_ner=False):
        """
        Augment document content by extracting metadata from document ocntent, namely Named-Entity and Topic information
        Args:
            documents:
            topics_config:
            ner_config:
            run_topics:
            run_ner:
            overwrite_topics:
            overwrite_ner:

        Returns:

        """
        # entities
        if ner_config and run_ner:
            logging.info("Extracting entities")
            ner_config = read_json_str_or_file(ner_config)
            for document in documents:
                if skip_content_insertion("entities", document, can_overwrite=overwrite_topics):
                    continue
                entities = get_named_entities(doc_id=document["id"], content=document["content"], ner_config=ner_config)
                document['annotations']['entities'] = entities

        # topics
        if topics_config and run_topics:
            topics_config = read_json_str_or_file(topics_config)
            topic_model = TopicModel(topics_config, embedder=embedder)
            logging.info("Extracting topics")
            for doc_idx, document in enumerate(documents):
                if skip_content_insertion("topics", document, can_overwrite=overwrite_ner):
                    continue
                logging.info(f"Getting topics for doc {doc_idx + 1}/{len(documents)}: {document['link']}")
                document["topics"] = topic_model.get_topics(content=document["content"])
        # superfluous return
        return documents

    def get_inputs(self, input_type, elastic_config, newline_norm=False, output_folder="retrieved_docs",
                   input_path=None,
                   input_title=None):
        """
        Args:
            input_type: JSON string or path to disk
            input_path: Path for input instance file
            input_title: Input title string
            elastic_config: JSON elastic configuration
            newline_norm:  whether to apply newline normalization
            output_folder: Where to save retrieved documents

        Returns:
            The input documents

        """
        pipeline_already_ran = False
        if input_type == "output":
            logging.info(f"Loading existing pipeline outputs from path {input_path}")

            # fetch existing pipeline outputs from local disk JSON file
            with open(input_path) as f:
                data = json.load(f)
            docs = data["documents"]
            pipeline_already_ran = True

        elif input_type == "elastic":
            # fetch from elasticsearch
            if self.es_retrieve is None:
                self.es_retrieve = ElasticSearchConfig(elastic_config)
            retrieve_args = elastic_config['retrieval']
            docs = self.es_retrieve.retrieve_documents(**retrieve_args)
            search_id = self.es_retrieve.get_last_search_id()

            # save retrieved docs
            os.makedirs(output_folder, exist_ok=True)
            with open(join(output_folder, f"docs_{len(docs)}_{search_id}.json"), "w") as f:
                logging.info(f"Saving obtained docs from elastic to {f.name}")
                json.dump(docs, f, ensure_ascii=False)

            docs = Pipeline.postprocess_input_docs(docs, newline_norm=newline_norm)

        elif input_type == "instance":
            input_instance = input_path
            if os.path.exists(input_instance):
                logging.info(f"Reading input instance text from {input_instance}")
                with open(input_instance) as f:
                    text = f.read()
                title = input_title
                docs = [
                    {
                        "id": make_segment_id(),
                        "content": text,
                        "title": title
                    }
                ]
            else:
                # assume it's JSONic input
                try:
                    docs = json.loads(input_instance)
                    if not isinstance(docs, list):
                        docs = [docs]
                    for i, doc in enumerate(docs):
                        if type(doc) is str:
                            # single stirng input -- assume it's the text
                            docs[i] = Document.make_dummy_document(doc)

                except json.JSONDecodeError:
                    docs = [
                        {
                            "id": make_segment_id(),
                            "content": input_instance,
                            "title": "Τίτλος κειμένου"
                        }
                    ]

            # treat is as a literal input text

            # template
            # {
            #     "link": "https://www.cretalive.gr/kriti/giati-ohi-aerodromio-sto-kastelli-anoihti-epistoli-apo-paratiritirio-politon",
            #     "item_type": "article",
            #     "feed_type": "crawl.cretalive",
            #     "redirect_url": "https://www.cretalive.gr/kriti/giati-ohi-aerodromio-sto-kastelli-anoihti-epistoli-apo-paratiritirio-politon",
            #     "title": "the tile"
            #     "content": "the content"
            #     "category": [],
            #     "tags": [
            #         "ΡΕΘΥΜΝΟ",
            #         "ΧΑΝΙΑ",
            #         "ΛΑΣΙΘΙ",
            #         "HAPPYNEWS",
            #         "ΗΡΑΚΛΕΙΟ"
            #     ],
            #     "publishedAt": "2020-12-14T21:30:33Z",
            #     "crawledAt": "2020-12-14T21:30:33Z",
            #     "date": "2020-12-14T21:30:33Z",
            #     "top_image": "https://www.cretalive.gr/sites/default/files/styles/og_image/public/migration/2017-07/3232232444.jpg?itok=PPGua75n",
            #     "domain": "cretalive",
            #     "netloc": "www.cretalive.gr",
            #     "type": "cretalive.article",
            #     "id": "65bb74d6d2397e6ed670dacb0e63a29134df0569"
            # }

            docs = Pipeline.postprocess_input_docs(docs, newline_norm=newline_norm)

        else:
            raise NotImplementedError(f"Undefined input type: {input_type}")
        return docs, pipeline_already_ran

    @staticmethod
    def postprocess_input_docs(docs, newline_norm=False):
        # post-processing
        if newline_norm:
            logging.info(f"Normalizing newlines")
            for doc in docs:
                doc['content'] = normalize_newlines(doc['content'])
        return docs

    @staticmethod
    def run_pipeline(documents, adu_model, rel_model, stance_model, cleanup_memory=True, run_rel=True, run_stance=True):
        """
        Function to execute the DebateLab pipeline.

            | 1. Loads the documents from the SocialObservatory Elasticsearch
            | 2. Runs the argument mining & cross-document clustering pipelines
            | 3. Validates the json output objects & performs corrections
            | 4. Stores the results in the DebateLab Elasticsearch
            | 4. Notifies ICS
        """
        # add version. Place it on the top level of the doc since the model may affect more than annotations (e.g. topics)
        for doc in documents:
            doc["model_version"] = str(MODEL_VERSION)
        # run ADU operations
        # get major_claims, claims, premises
        logging.info("Running ADU extraction.")
        adu_results = adu_pipeline.run(documents, adu_model)
        if cleanup_memory:
            adu_model.empty_cuda_cache()
            del adu_model

        # update docs
        for doc_idx, (doc, segs) in enumerate(zip(documents, adu_results)):
            # set model version
            # convert IDs
            doc['annotations'] = {"ADUs": [], "Relations": []}
            mcs = list(segs["major_claims"])
            assert len(mcs) <= 1, "Multiple major claims after pipeline run!"
            if not mcs:
                logging.warning(
                    f"Document: {doc_idx + 1}/{len(documents)}: {doc['id']} has no major claims -- segment info: {segs} - skipping ID assignment.")
                continue
            mcs[0]["id"] = "T1"
            nonmcs = [y for k in ("claims", "premises") for y in segs[k]]
            # order by starting point
            nonmcs = sorted(nonmcs, key=lambda x: x["starts"])
            # assign ids
            for i, s in enumerate(nonmcs):
                s["id"] = f"T{i + 2}"
            doc['annotations']["ADUs"] = mcs + nonmcs

        # run rel
        logging.info(f"{'Running' if run_rel else 'Skipping disabled'} REL classification.")
        if run_rel:
            # logging.info("Running REL classification.")
            rel_results = rel_pipeline.run(rel_model, adu_results)
            if cleanup_memory:
                rel_model.empty_cuda_cache()
                del rel_model

            # update docs
            for doc, rels in zip(documents, rel_results):
                # convert IDs
                for i, r in enumerate(rels):
                    r["id"] = f"R{i + 1}"
                doc['annotations']['Relations'] = rels

        # run stance
        logging.info(f"{'Running' if run_stance else 'Skipping disabled'} STANCE classification.")
        if run_stance:
            stance_results = stance_pipeline.run(stance_model, adu_results)
            if cleanup_memory:
                stance_model.empty_cuda_cache()
                del stance_model

            # update docs
            for doc, sts in zip(documents, stance_results):
                adus = doc['annotations']['ADUs']
                mcs = [x for x in adus if x['type'] == "major_claim"]
                claims = [x for x in adus if x['type'] == "claim"]
                assert len(mcs) <= 1, f"Found {len(mcs)} MCs for doc {doc['id']} in stance result processing!"
                assert len(claims) == len(
                    sts), f"Found {len(sts)} stance results for doc {doc['id']} in stance result processing!"
                for i, v in enumerate(sts.values()):
                    assert len(v) == 1, "Multiple-stance claim detected!"
                    v[0]['id'] = f"A{i + 1}"

                for claim_id_in_stance in sts:
                    claim_dicts = [x for x in doc['annotations']['ADUs'] if x['id'] == claim_id_in_stance]
                    try:
                        assert len(claim_dicts) == 1, f"Multiple stance annotations in claim: {claim_dicts} "
                    except AssertionError:
                        print()

                    claim_dicts[0]["stance"] = sts[claim_id_in_stance]

        logging.info("Mining pipeline complete.")
        return documents

    @staticmethod
    def prepare_outputs(arguments, documents, valid_document_ids):
        # mark validity and save outputs
        for doc_idx, doc in enumerate(documents):
            doc["valid"] = int(doc["id"] in valid_document_ids)

            majcount = 0
            for adu in doc['annotations']['ADUs']:
                s, e = int(adu["starts"]), int(adu["ends"])
                adu['confidences'] = [str(x) for x in adu['confidences']]
                if doc['content'][s:e] != adu["segment"]:
                    logging.error(
                        f"ERROR: Mismatch between offsets and segment {doc['id']}")
                    logging.error("DOC:", doc['content'][s:e])
                    logging.error("ADU:", adu['segment'])
                    raise ValueError("Segment-content mismatch!")
                if adu['type'] == 'major_claim':
                    majcount += 1
            if majcount > 1:
                raise ValueError(
                    f"ERROR: got {majcount} major claims for document {doc['id']}")

            if arguments.strip_document_extra_info:
                documents[doc_idx] = strip_document_extra_info(doc)

        outputs = {"documents": documents, "arguments": vars(arguments)}
        return outputs

    def cleanup(self):
        """
        ES Endpoint cleanup function
        """
        if not self.persistent_run:
            for es_endpoint in (self.es_retrieve, self.es_save):
                if es_endpoint is not None:
                    es_endpoint.stop()


if __name__ == "__main__":
    args = get_args()

    config_dict = vars(args)
    pipeline = Pipeline(vars(args))
    config = pipeline.update_configuration(config_dict)
    pipeline.run(config)
    logging.info("Main done, bye!")
