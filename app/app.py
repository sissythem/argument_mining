import logging
import uvicorn
from os import getcwd
from os.path import join
from logging.config import dictConfig

from fastapi import FastAPI
from fastapi.logger import logger

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)

app = FastAPI()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=["POST"])
def predict():
    # title = request.args.get("title")
    with open(join(getcwd(), "resources", "example.json"), "r") as f:
        return f.read(), 200


if __name__ == "__main__":
    uvicorn.run(app)
