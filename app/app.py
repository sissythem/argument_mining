import logging
from os import getcwd
from os.path import join
from logging.config import dictConfig

from flask import Flask  # , request

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    },
})
app = Flask(__name__)
logger = app.logger

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=["POST"])
def predict():
    # title = request.args.get("title")
    with open(join(getcwd(), "resources", "example.json"), "r") as f:
        return f.read(), 200
