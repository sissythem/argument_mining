import json
from os import getcwd
from os.path import join
from typing import AnyStr, List, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.config import AppConfig

app = FastAPI()
config = AppConfig()
logger = config.app_logger


class PipelineRequest(BaseModel):
    links: Union[List[AnyStr], AnyStr]


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

    # TODO remove demo code below
    app_path = getcwd()
    if app_path.endswith("mining"):
        app_path = join(app_path, "app")
    with open(join(app_path, "resources", "example.json"), "r") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    uvicorn.run(app)
