import logging
from fastapi import FastAPI, HTTPException, Request


def log_return_exception(msg: str, code: int = 400):
    logging.error(msg)
    raise HTTPException(status_code=500, detail=msg)
