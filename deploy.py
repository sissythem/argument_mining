import json
import os
import sys
from os.path import join, isdir, basename
from typing import AnyStr, Dict, List, Union, Optional
from fastapi_utils.tasks import repeat_every

from fastapi import Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader, APIKey
import requests
import uvicorn
import copy
import glob
import argparse
from copy import deepcopy
import logging
from src.training_api import ModelLoader
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pipeline import Pipeline
from src.io.elastic_io import ElasticSearchConfig
from src.crawling import CrawlerProber
from src.pipeline.crossdoc import write_additional_cross_doc_inputs
from src.annotation import read_promising_docs_for_annotation, empty_promising_docs_for_annotation
from src.utils import timestamp, read_json_str_or_file

from src.api_utils import log_return_exception
from src.language import LangDetect as LangDetector
import subprocess

from pipeline import DEFAULT_ADDITIONAL_CROSSDOCS_PATH
from train import run as run_training

deploy_parser = argparse.ArgumentParser()
deploy_parser.add_argument("-device", help="Number of gpus", default="cpu")
deploy_parser.add_argument("-port", help="Deployment port", default=8000)
deploy_parser.add_argument("-disable_authentication", help="Whether to use authentication", default=False,
                           action='store_true')
deploy_parser.add_argument("-trained_models", help="Path to json file with trained models information",
                           default="trained_models_info.json")
deploy_parser.add_argument("-n_gpu", help="Number of gpus", default=1)
deploy_parser.add_argument("-models_base_path", help="Path to folder with available models")
deploy_parser.add_argument("-crawler_endpoint", help="Web crawler endpoint")
deploy_parser.add_argument("-additional_cross_doc_endpoint", help="Web crawler endpoint")
deploy_parser.add_argument("--allow_multiple_models", help="Whether to allow for model training and id selection.",
                           action="store_true", default=False)
deploy_parser.add_argument("-crawler_credentials_path",
                           help="File with crawler credentials, in the form of username<newline>password")
deploy_parser.add_argument("-pipeline_configuration", help="Pipeline configuration json.")
deploy_parser.add_argument("-write_crossdocs_to", help="Where to write pipeline output docs as additional crossdocs",
                           choices=["disk", "object", "url"], default=None)
deploy_parser.add_argument("-crossdoc_output_path", help="", default=DEFAULT_ADDITIONAL_CROSSDOCS_PATH)

args = deploy_parser.parse_args()

API_KEY_NAME = "access_token"
AUTHORIZED_KEYS = "./authorized_keys"

TRAINING_LOCKS_FOLDER = 'training_locks'
os.makedirs(TRAINING_LOCKS_FOLDER, exist_ok=True)

#####
logging.getLogger().setLevel(logging.DEBUG)
logging.info("Starting FastAPI")
app = FastAPI()
language_detector = LangDetector()

if not args.disable_authentication:
    if os.path.exists(AUTHORIZED_KEYS):
        with open(AUTHORIZED_KEYS) as f:
            authorized_keys = json.load(f)
        logging.info("Loaded keys:")
        for k, v in authorized_keys.items():
            logging.info(f"Loaded {len(v)} [{k}] keys.")
        logging.info("----")
    else:
        logging.error("Could not load authorized keys!")
        authorized_keys = {"user": [], "admin": []}

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)


# async def get_user_api_key(api_key_query: str = Security(api_key_query)):


async def get_user_api_key(api_key_query: str = Security(api_key_query)):
    return get_api_key(api_key_query, "user")


async def get_admin_api_key(api_key_query: str = Security(api_key_query)):
    return get_api_key(api_key_query, "admin")


def get_api_key(api_key_query: str = Security(api_key_query), query_type: str = "user"):
    try:
        if args.disable_authentication:
            return api_key_query
        if api_key_query in authorized_keys[query_type]:
            return api_key_query
    except Exception as ex:
        log_return_exception(f"Unknown error: {ex}.", 500)
    except KeyError:
        log_return_exception(f"Undefined key query of type {query_type}.", 500)
    log_return_exception(
        f"Invalid / Missing authentication: Need to specify valid access token with [{query_type}] privileges via the [{API_KEY_NAME}] parameter.",
        403)


pipeline = None
DEFAULT_MODEL_ID = "configured"
model_loader = ModelLoader(args.models_base_path)

if args.pipeline_configuration:
    if os.path.exists(args.pipeline_configuration):
        with open(args.pipeline_configuration) as f:
            configuration = json.load(f)
    else:
        configuration = json.loads(args.pipeline_configuration)
    logging.info("Creating pipeline...")
    pipeline = Pipeline(configuration, persistent_run=True)
    model_loader.add_model_id(DEFAULT_MODEL_ID, {"adu": configuration['adu_model'],
                                                 "rel": configuration['rel_model'],
                                                 "stance": configuration['stance_model'],
                                                 "embedder": configuration['embedder_model']
                                                 })
    if not args.allow_multiple_models:
        # load models once, here
        pipeline.load_store_models(model_loader.get_model_config(DEFAULT_MODEL_ID), store=True)

    with open(configuration['elastic_config']) as f:
        elastic_config = json.load(f)

    es_retriever = ElasticSearchConfig(elastic_config["retrieve"])
    pipeline.es_retrieve = es_retriever
    crawler_prober = CrawlerProber(args.crawler_credentials_path, es_retriever)

# load available models
model_loader.load_available_models(args.models_base_path)


class TrainRequest(BaseModel):
    config: str = "{}"
    data: str = None
    type: str = "adu"
    id: str = f"api_training_{timestamp()}"


class PipelineRequest(BaseModel):
    """
    Request for the prediction API

    Arguments:
        links: URLs to ingest
        texts: Raw text to ingest
        model_id: The id of the model to utilize for the prediction
        config: Settings to override default behaviour
        use_default_missing_models: Whether to use default components for those that are missing for the selected model id
    """
    links: Union[List[AnyStr], AnyStr] = []
    texts: Union[List[AnyStr], AnyStr] = []
    model_id: str = None
    config: Dict = {}
    use_default_missing_models: bool = True


class CrawlRequest(BaseModel):
    links: Union[List[AnyStr], AnyStr] = []


class CrossDocRequest(BaseModel):
    docs: List = []


class PromisingDeleteRequest(BaseModel):
    doc_ids: Union[List[AnyStr], List[Dict]] = None


class ValidationException(Exception):
    def __init__(self, message: AnyStr, document: Dict, validation_errors: List):
        self.message = message
        self.document = document
        self.validation_errors = validation_errors


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    logging.error(f"Raising exception for request: {request}")
    return JSONResponse(
        status_code=500,
        content={"message": exc.message, "document": exc.document, "validation_errors": exc.validation_errors},
    )


@app.on_event("startup")
async def startup_event():
    pass


@app.get('/')
def hello_world():
    return 'Hello World!'


@app.get('/load_models')
def load_models(api_key: APIKey = Depends(get_admin_api_key)):
    if pipeline is None:
        log_return_exception(f"Attempted to load models for non-instantiated pipeline.", 500)

    pipeline.load_store_models(model_loader.get_model_config())
    return JSONResponse(
        status_code=200,
        content={"message": "Successfully loaded models."},
    )


@app.post('/crossdocs')
async def submit_crossdocs(crossdoc: CrossDocRequest,
                           api_key: APIKey = Depends(get_user_api_key)
                           ):
    docs = crossdoc.docs
    if not docs:
        log_return_exception("Missing [docs] field in the request request payload.", 400)
    try:
        await _submit_crossdocs(docs, auth_params={"access_token": api_key})
    except Exception as ex:
        log_return_exception(f"Uncaught exception : {ex}", 500)


async def _submit_crossdocs(docs: list, auth_params=None):
    logging.info(f"Submitting {len(docs)} crossdocs to {args.write_crossdocs_to}.")
    if args.write_crossdocs_to == "disk":
        wrote = write_additional_cross_doc_inputs(args.crossdoc_output_path, docs)
        logging.info(
            f"Wrote {len(wrote)} new documents for future cross-doc clustering to disk: {args.crossdoc_output_path}")

    elif args.write_crossdocs_to == "object":
        await pipeline.insert_crossdocs(docs)
        logging.info(
            f"Inserted {len(docs)} new documents for future cross-doc clustering, total now: {len(pipeline.additional_crossdocs)}")
    elif args.write_crossdocs_to == "url":
        logging.info(f"Submitted {len(docs)} via POST to: {args.crossdoc_output_path}")
        requests.post(args.crossdoc_output_path, json={"docs": docs}, params=auth_params)


@app.get('/promising')
async def get_promising_docs():
    docs = read_promising_docs_for_annotation()
    logging.info(f"Fetching {len(docs)} promising docs.")
    return docs


@app.delete('/promising')
async def delete_promising_docs(request: PromisingDeleteRequest,
                                api_key: APIKey = Depends(get_user_api_key)
                                ):
    try:
        ids = request.doc_ids
        for i, id_ in enumerate(ids):
            if isinstance(id_, str):
                ids[i] = {"id": id_}
            elif isinstance(id_, list) and len(id_) == 2:
                # id, model_version
                id_, model_version = id_
                ids[i] = {"id": id_, "model_version": model_version}
        logging.info(f"Submitting deletion for {len(ids) if ids is not None else '<none>'} promising docs")
        empty_promising_docs_for_annotation(docs_to_delete=request.doc_ids)
    except Exception as ex:
        log_return_exception(f"Uncaught exception : {ex}")


@app.post('/crawl')
async def crawl(crawl_request: CrawlRequest):
    links = crawl_request.links
    if not links:
        log_return_exception("Missing [url] field in the request request payload.", 400)
    if type(links) == str or type(links) == bytes:
        links = [links]
    links = [link.decode('utf8') if type(link) == bytes else link for link in links]
    docs, fail_messages = [], []
    for link in links:
        doc, fail_msg = crawler_prober.probe(link)
        if fail_msg:
            fail_messages.append(fail_msg)
        if doc is None:
            continue
        docs.append(doc)
    return docs


@app.post('/train')
def train(train_request: TrainRequest,
          training_background_task: BackgroundTasks,
          api_key: APIKey = Depends(get_user_api_key),
          ):
    """
    ADU training endpoint
    Args:
        train_request:
        api_key:

    Returns:

    """
    try:
        # submit for training
        if model_loader.is_in_progress(train_request.id, train_request.type):
            log_return_exception(f"Model id [{train_request.id}] already currently in training.")

        # lock_path = join(TRAINING_LOCKS_FOLDER, f"{train_request.type}_{train_request.id}")
        # with LockContext(lock_path, timeout=5):
        if model_loader.has_id(train_request.id):
            log_return_exception(f"Model id [{train_request.id}] already exists in the model repository.", 400)

        status_file_path = join(args.models_base_path, train_request.id, train_request.type, "status")
        # if lock_read_file(status_file_path)

        # write training data to disk
        local_train_data_path = "rest_training_data"
        data = train_request.data
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        ndata = len(data.split("\n"))
        logging.info(f"Parsed {ndata} instances from csv input")
        with open(local_train_data_path, "w") as f:
            logging.info(f"Writing training data to path: [{f.name}]")
            f.writelines(data)
        # training args
        train_hyperparams = {
            "batch_size": 32,
            "num_epochs": 80,
            "strategy": "epoch",
            "eval_steps": 5
        }
        for k, v in json.loads(train_request.config).items():
            train_hyperparams[k] = v
        output_path = join(args.models_base_path, train_request.id, train_request.type)
        train_args = {
            "-train_data": local_train_data_path,
            "-test_data": local_train_data_path,
            "-train_args": json.dumps(train_hyperparams),
            "--model_output_path": output_path,
            "--status_file": status_file_path
        }

        # adu
        model_type = "adu"
        cargs = [x for (k, v) in train_args.items() for x in (k, v)]
        cmd = [sys.executable, "train.py", "adu"] + cargs
        if model_loader.can_start_training():
            logging.info(f"Submitting model [{train_request.id}] for training.")
            training_background_task.add_task(submit_training, train_request.id, cmd, model_type, output_path)
        else:
            log_return_exception(
                f"Cannot start training, since max number of concurrent sessions are: {model_loader.max_concurrent_training}, existing sessions are: [{model_loader.summarize_in_progress()}].")
            # await submit_training(cmd, lock_path)
        logging.info(f"Submitted training model with id: [{train_request.id}]")
        return JSONResponse(
            status_code=200,
            content={"message": "Submitted for training successfully."}
        )
    except (json.JSONDecodeError) as ex:
        log_return_exception(f"Failed to invoke training: {ex}.")
    except ValueError as ve:
        log_return_exception(f"Exception when trying to begin training: {ve}.")


@app.on_event("startup")
@repeat_every(seconds=60 * 10)  # 10 mins
def check_training_status():
    model_loader.check_models_in_progress()


# iterate model(s) in training


async def submit_training(request_id, command, model_type, model_path):
    logfile = glob.glob(join(model_path, "*", "*.log"))
    model_loader.start_training(request_id, model_type)
    print(f"Spawning... {request_id} {model_type}")
    subprocess.Popen(command)


@app.get('/models')
async def get_models(api_key: APIKey = Depends(get_user_api_key)):
    return JSONResponse({
        "available": model_loader.ids_to_paths,
        "in_progress": model_loader.in_progress,
        "currently selected": model_loader.get_model_config()
    })


@app.post('/predict')
async def predict(pipeline_request: PipelineRequest,
                  api_key: APIKey = Depends(get_user_api_key)):
    if pipeline is None:
        log_return_exception(f"Attempted to generate prediction from for non-instantiated pipeline.")

    logging.info(f"Running prediction API with request: {str(pipeline_request)}")
    links, texts = pipeline_request.links, pipeline_request.texts
    if not links and not texts:
        log_return_exception(
            "Missing both [url] and [text] fields in the request request payload -- need at least one.", 400)

    try:
        # lock pipeline
        if args.allow_multiple_models:
            if pipeline.is_in_use():
                log_return_exception(f"Pipeline is occupied -- try again later.")
            pipeline.set_in_use()

            # load requested model id
            model_id = pipeline_request.model_id or model_loader.selected_model
            logging.info(
                f"Using multimodel prediction api with {'the default' if model_id == DEFAULT_MODEL_ID else ''} model id: {model_id}")
            if model_loader.has_id(model_id):
                model_config = copy.deepcopy(model_loader.get_model_config(model_id))
            else:
                if model_loader.is_in_progress(model_id):
                    log_return_exception(
                        f"Requested model id: {model_loader.get_model_in_training(model_id)} is currently in training:",
                        501)
                else:
                    log_return_exception(
                        f"Undefined model id: {model_id}, available ones are: {model_loader.summarize_available()}",
                        501)

            # default components
            default_components = []
            if pipeline_request.use_default_missing_models:
                for model_type, default_model_path in model_loader.ids_to_paths[DEFAULT_MODEL_ID].items():
                    if model_type not in model_config:
                        logging.info(
                            f"Falling back to the default id [{DEFAULT_MODEL_ID}] for model type [{model_type}] from {default_model_path}.")
                        model_config[model_type] = default_model_path
                    else:
                        logging.info(
                            f"Using the requested id: [{model_id}] for type {model_type} from path [{model_config[model_type]}].")
            for model_type, path in model_config.items():
                pipeline.load_model(path, model_type, args.device, store=True)

            logging.info(
                f"Using requested model [{model_id}] for supported components: {list(model_loader.get_model_config(model_id).keys())}")

        if isinstance(links, str) or isinstance(links, bytes):
            links = [links]
        if isinstance(texts, str) or isinstance(texts, bytes):
            texts = [texts]
        links = [link.decode('utf8') if type(link) == bytes else link for link in links]
        texts = [text.decode('utf8') if type(text) == bytes else text for text in texts]

        # pass configuration
        config = deepcopy(configuration)
        for k, v in pipeline_request.config.items():
            logging.info(f"Setting API parameter: {k} = {v}")
            config[k] = v

        config = pipeline.update_configuration(config)
        fail_messages = []
        text_results, link_results = {"documents": [], "cross_doc_relations": []}, {"documents": [],
                                                                                    "cross_doc_relations": []}
        # links
        if links:
            inputs = []
            for link in links:
                doc, fail_msg = crawler_prober.probe(link)
                if fail_msg:
                    fail_messages.append(fail_msg)
                if doc is None:
                    continue

                lang = language_detector.detect(doc["content"])
                if not language_detector.is_greek(language=lang):
                    fail_messages.append(f"Skipping article {link} due to non-greek detected language: {lang}")
                    continue

                inputs.append(doc)
            if inputs:
                # run pipeline with the link inputs
                config.input_type = "instance"
                config.input_path = json.dumps(inputs)

                link_results = pipeline.run(config)

        if texts:
            inputs = []
            config.input_type = "instance"
            for t, text in enumerate(texts):
                lang = language_detector.detect(text)
                if not language_detector.is_greek(language=lang):
                    fail_messages.append(
                        f"Skipping submitted text {t + 1 / len(texts)}: [{text[:20]}...] due to non-greek detected language: {lang}")
                    continue
                inputs.append(text)
            if inputs:
                config.input_path = json.dumps(inputs)
                text_results = pipeline.run(config)

        results = {
            "documents": text_results["documents"] + link_results["documents"],
            "cross_doc_relations": text_results["cross_doc_relations"] + link_results["cross_doc_relations"]
        }
        logging.info(
            f"Arg.mining API returning {len(results['documents'])} results out of {len(links)} url and {len(texts)} text input requests.")

        if args.write_crossdocs_to is not None:
            docs = results["documents"]
            if docs:
                try:
                    await _submit_crossdocs(docs, auth_params={"access_token": api_key})
                except Exception as ex:
                    fail_msg = f"Failed to POST additional docs for cross-doc clustering to {args.crossdoc_output_path}: {ex}"
                    logging.error(fail_msg)
                    fail_messages.append(fail_msg)

        return JSONResponse(
            status_code=200,
            content={"failures": fail_messages,
                     "message": "Successful pass" if not fail_messages else "Failure(s) occured.", **results}
        )
    finally:
        # unlock pipeline
        if args.allow_multiple_models:
            pipeline.release_use()


if __name__ == "__main__":
    logging.info("Running api...")
    uvicorn.run(app, host="0.0.0.0", port=int(args.port))
    print("test")
