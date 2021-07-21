import json
from os import getcwd
from os.path import join
from typing import AnyStr, Dict, List, Union

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils.config import AppConfig
from src.pipeline.debatelab import DebateLab
from src.pipeline.validation import JsonValidator

app = FastAPI()
config = AppConfig()
logger = config.app_logger
crawler_url = config.properties["eval"]["crawler_endpoint"]


class PipelineRequest(BaseModel):
    links: Union[List[AnyStr], AnyStr]


class ValidationException(Exception):
    def __init__(self, message: AnyStr, document: Dict, validation_errors: List):
        self.message = message
        self.document = document
        self.validation_errors = validation_errors


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    logger.error(f"Raising exception for request: {request}")
    return JSONResponse(
        status_code=500,
        content={"message": exc.message, "document": exc.document, "validation_errors": exc.validation_errors},
    )


@app.on_event("startup")
async def startup_event():
    # TODO train
    pass


@app.get('/')
def hello_world():
    return 'Hello World!'


@app.post('/predict')
def predict(pipeline_request: PipelineRequest):
    links = pipeline_request.links
    if not links:
        raise HTTPException(status_code=500, detail="Missing URLs in request body for downloading")
    if type(links) == str or type(links) == bytes:
        links = [links]
    links = [link.decode('utf8') if type(link) == bytes else link for link in links]
    link = links[0]
    data = {
        "spider_name": "article", "request": {
            "url": link}
    }
    try:
        response = requests.post(url=crawler_url, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Could not retrieve article: {response.text}")
    except(BaseException, Exception) as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve article: {e}")
    document = json.loads(response.text)
    debatelab = DebateLab(app_config=config)
    validator = JsonValidator(app_config=config)
    document, segment_counter, rel_counter, stance_counter = debatelab.predict(document=document)
    counters = {"adu": segment_counter, "rel": rel_counter, "stance": stance_counter}
    validation_errors, invalid_adus, corrected = validator.run_validation(document=document, counters=counters)
    if not validation_errors:
        json_compatible_item_data = jsonable_encoder(document)
        return JSONResponse(content=json_compatible_item_data, status_code=200)
    else:
        raise ValidationException(message="ADU & relation predictions did not pass validation", document=document,
                                  validation_errors=validation_errors)


if __name__ == "__main__":
    uvicorn.run(app)
