import json
from os import getcwd
from os.path import join

import uvicorn
from fastapi import FastAPI

from src.utils.config import AppConfig

app = FastAPI()
config = AppConfig()
logger = config.app_logger


@app.get('/')
def hello_world():
    return 'Hello World!'


@app.post('/predict')
def predict():
    # title = request.args.get("title")
    app_path = getcwd()
    if app_path.endswith("mining"):
        app_path = join(app_path, "app")
    with open(join(app_path, "resources", "example.json"), "r") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    uvicorn.run(app)
